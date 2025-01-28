from .subnet import *
from .subnet.bitEstimator import ICLR17EntropyCoder
gpu_num = torch.cuda.device_count()

class VideoCompressor(nn.Module):
    def __init__(self, train_lambda):
        super(VideoCompressor, self).__init__()
        self.addres = False
        self.Encoder = FeatureEncoder()
        self.Decoder = FeatureDecoder()
        self.aligner = PCD_Align()

        self.OffsetPriorEncoder = OffsetPriorEncodeNet()
        self.OffsetPriorDecoder = OffsetPriorDecodeNet()
        self.EntropyCoding_offsetf = NIPS18nocEntropyCoder()
        self.EntropyCoding_offsetz = ICLR17EntropyCoder(out_channel_mv)

        self.resEncoder = ResEncodeNet()
        self.resDecoder = ResDecodeNet()
        self.resPriorEncoder = ResPriorEncodeNet()
        self.resPriorDecoder = ResPriorDecodeNet()
        self.EntropyCoding_residualf = NIPS18nocEntropyCoder()
        self.EntropyCoding_residualz = ICLR17EntropyCoder(out_channel_resN)

        self.motionmse = 0
        self.finalmse = 1
        self.motionbpp = 1
        self.residualbpp = 1
        self.train_lambda = train_lambda
        self.true_lambda = self.train_lambda

    def Trainall(self):
        for p in self.parameters():
            p.requires_grad = True

    def TrainwoMotion(self):
        for p in self.parameters():
            p.requires_grad = True
        for p in self.OffsetPriorEncoder.parameters():
            p.requires_grad = False
        for p in self.OffsetPriorDecoder.parameters():
            p.requires_grad = False
        for p in self.EntropyCoding_offsetf.parameters():
            p.requires_grad = False
        for p in self.EntropyCoding_offsetz.parameters():
            p.requires_grad = False
        for p in self.aligner.parameters():
            p.requires_grad = False


    def Trainstage(self, global_step):
        if global_step < 200000:
            self.motionmse = 1
            self.motionbpp = 1
            self.residualbpp = 0
            self.finalmse = 0
        elif global_step < 400000:
            self.TrainwoMotion()
            self.motionmse = 0
            self.motionbpp = 0
            self.residualbpp = 0
            self.finalmse = 1
        elif global_step < 500000:
            self.TrainwoMotion()
            self.motionmse = 0
            self.motionbpp = 0
            self.residualbpp = 1
            self.finalmse = 1
        else:
            self.Trainall()
            self.motionmse = 0
            self.motionbpp = 1
            self.residualbpp = 1
            self.finalmse = 1


    def Q(self, x):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        else:
            return torch.round(x)

    def AlignMultiFeature(self, ref_feature, input_feature):
        # motion estimation + motion compression czyli kodowanie wartości cech wartości optical flow )
        en_offset = self.aligner.MotionEstimation(ref_feature, input_feature)
        # Tensor en_offset nie jest reshapowany, wiec ma shape B,128,16,16 ale to jest rownowazne tensorowi B,2,128,128, czyli optical flow
        # jeszcze bardziej zmniejsza rozmiar przestrzenny z 16x16 do 4x4, ale nie zwiększa liczby cech (nadal 128)
        encoded_offset_prior = self.OffsetPriorEncoder(en_offset)
        # kwantyzacja priorów i estymacja bitów tych priorów
        q_offset_prior, bits_offsetz = self.EntropyCoding_offsetz(encoded_offset_prior)
        # powrot do 128x16x16 - sigmy
        decoded_offset_prior = self.OffsetPriorDecoder(q_offset_prior)

        # kwantyzacja offsetow
        q_offset, bits_offsetf = self.EntropyCoding_offsetf(en_offset, decoded_offset_prior)

        aligned_feature = self.aligner.MotionCompensation(ref_feature, q_offset)


        return aligned_feature, bits_offsetf, bits_offsetz

    def AddResidual(self, input_feature, aligned_feature):
        res_feature = input_feature - aligned_feature
        encoded_residual = self.resEncoder(res_feature)

        # hyperprior
        encoded_residual_prior = self.resPriorEncoder(encoded_residual)
        q_encoded_residual_prior, bits_residualz = self.EntropyCoding_residualz(encoded_residual_prior)
        decoded_residual_prior = self.resPriorDecoder(q_encoded_residual_prior)

        q_encoded_residual, bits_residualf = self.EntropyCoding_residualf(encoded_residual, decoded_residual_prior)
        output_feature = aligned_feature + self.resDecoder(q_encoded_residual)

        return output_feature, bits_residualf, bits_residualz

    def GetLoss(self, input_image, recon_image, aligned_image, input_recon, bits_offsetf, bits_offsetz, bits_residualf, bits_residualz):
        out = dict()

        out["mse_loss"] = torch.mean((recon_image - input_image).pow(2))
        out["input_loss"] = torch.mean((input_recon - input_image).pow(2))
        out["align_loss"] = torch.mean((aligned_image - input_image).pow(2))
        im_shape = input_image.size()
        allarea = im_shape[0] * im_shape[2] * im_shape[3]

        out["bpp_offsetf"] = bits_offsetf / allarea
        out["bpp_offsetz"] = bits_offsetz / allarea
        out["bpp_residualf"] = bits_residualf / allarea
        out["bpp_residualz"] = bits_residualz / allarea
        out["bpp"] = out["bpp_offsetf"] + out["bpp_offsetz"] + out["bpp_residualf"] + out["bpp_residualz"]
        out["rd_loss"] = (
                self.true_lambda * (self.finalmse *out["mse_loss"] + self.motionmse * out["align_loss"] + 0.1 * out["input_loss"])
                + self.motionbpp * (out["bpp_offsetf"] + out["bpp_offsetz"])
                + self.residualbpp * (out["bpp_residualf"] + out["bpp_residualz"])
                )

        return out

    def forward(self, ref_image, input_image):
        # from pixel space to feature space
        ref_feature = self.Encoder(ref_image)
        input_feature = self.Encoder(input_image)

        input_recon = self.Decoder(input_feature)

        # feature space deformable compensation
        aligned_feature, bits_offsetf, bits_offsetz = self.AlignMultiFeature(ref_feature, input_feature)

        # feature space residual compression
        recon_feature, bits_residualf, bits_residualz = self.AddResidual(input_feature, aligned_feature)

        recon_image = self.Decoder(recon_feature)
        align_image = self.Decoder(aligned_feature)

        if not self.training:
            recon_image = recon_image.clamp(0., 1.)
            align_image = align_image.clamp(0., 1.)

        # calculating loss fucntion
        out = self.GetLoss(input_image, recon_image, align_image, input_recon, bits_offsetf, bits_offsetz, bits_residualf, bits_residualz)

        return recon_image, out
