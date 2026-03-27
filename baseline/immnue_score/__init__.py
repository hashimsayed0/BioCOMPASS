from .PD1 import PD1
from .PDL1 import PDL1
from .CTLA4 import CTLA4
from .GeneBio import GeneBio
from .Kong_NetBio import Kong_NetBio

from .Wu_MIAS import Wu_MIAS
from .Cristescu_GEP import Cristescu_GEP

from .Auslander_IMPRES import Auslander_IMPRES

from .Jiang_TIDE import Jiang_TIDE

from .Huang_NRS import Huang_NRS
from .Ayers_IFNG import Ayers_IFNG

from .CD8 import CD8
from .Davoli_CIS import Davoli_CIS
from .Roh_IS import Roh_IS
from .Fehrenbacher_Teff import Fehrenbacher_Teff


from .Jiang_CTLs import Jiang_CTLs
from .Jiang_TAMs import Jiang_TAMs
from .Jiang_Texh import Jiang_Texh


from .Messina_CKS import Messina_CKS
from .Nurmik_CAFs import Nurmik_CAFs

from .Rooney_ICA import Rooney_ICA
from .Freeman_PGM import Freeman_PGM


immnue_score_methods = {'PD1':PD1, 'PDL1':PDL1, 'CTLA4':CTLA4,  'CD8':CD8,  'GeneBio':GeneBio,  'NetBio':Kong_NetBio,
                        'MIAS': Wu_MIAS, 'GEP':Cristescu_GEP, 'IMPRES':Auslander_IMPRES, 'TIDE':Jiang_TIDE,
                        'NRS': Huang_NRS, 'IFNG':Ayers_IFNG, 'CIS': Davoli_CIS, 'IS': Roh_IS, 'Teff': Fehrenbacher_Teff, 'PGM':Freeman_PGM,
                        'CKS': Messina_CKS, 'CAF': Nurmik_CAFs, 'CTL': Jiang_CTLs, 'TAM': Jiang_TAMs, 'Texh':Jiang_Texh, 'ICA':Rooney_ICA}



