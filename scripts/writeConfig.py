import lsst.daf.persistence as dafPersist
import pfs.instmodel.makePfsConfig as makeConfig
import numpy as np
from pfs.drp.stella.datamodel.pfsConfig import PfsConfig as StellaPfsConfig

design_id = 123
# design_id = 0xfedcba9876543210

# This demonstrates that using a negative hash 
# (such as can be generated 50% of the time for a signed int)
# leads to wrong filename formatting through the butler 
# and even failure
# design_id = -123

visit0 = 456
date_obs = '2020-11-11'

d = makeConfig.makePfsDesign(design_id, np.array([1]), np.array([1]), np.array([1]), np.array([1]), np.array([1]))

# This doesn't work - as class method in pfs.datamodel.PfsConfig doesn't use cls object to instantiate
# c = StellaPfsConfig.fromPfsDesign(d, visit0, d.pfiNominal)

# Instead use full constructor
c = StellaPfsConfig(d.pfsDesignId, visit0, d.raBoresight, d.decBoresight, d.fiberId, d.tract, d.patch, d.ra, d.dec, d.catId, d.objId,
                    d.targetType, d.fiberStatus, d.fiberMag, d.filterNames, d.pfiNominal, d.pfiNominal)


dataId={'visit': visit0, 'pfsDesignId': design_id, 'dateObs': date_obs }

butler = dafPersist.Butler('.')
butler.put(c, 'pfsConfig', dataId)

