from weighting.abstract_weighting import AbsWeighting
from weighting.EW import EW
from weighting.GradNorm import GradNorm
from weighting.MGDA import MGDA
from weighting.UW import UW
from weighting.DWA import DWA
from weighting.GLS import GLS  # Unsupported log_grads
from weighting.GradDrop import GradDrop  # Unsupported log_grads
from weighting.PCGrad import PCGrad
from weighting.GradVac import GradVac
from weighting.IMTL import IMTL
from weighting.CAGrad import CAGrad
from weighting.Nash_MTL import Nash_MTL  # Unsupported log_grads
from weighting.RLW import RLW
from weighting.MoCo import MoCo  # Unsupported log_grads
from weighting.Aligned_MTL import Aligned_MTL
from weighting.DB_MTL import DB_MTL

__all__ = ['AbsWeighting',
           'EW', 
           'GradNorm', 
           'MGDA',
           'UW',
           'DWA',
           'GLS',
           'GradDrop',
           'PCGrad',
           'GradVac',
           'IMTL',
           'CAGrad',
           'Nash_MTL',
           'RLW',
           'MoCo',
           'Aligned_MTL',
           'DB_MTL']