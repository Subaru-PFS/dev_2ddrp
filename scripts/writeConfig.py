import lsst.daf.persistence as dafPersist
import pfs.instmodel.makePfsConfig as makeConfig
import numpy as np
from pfs.drp.stella.datamodel.pfsConfig import PfsConfig as StellaPfsConfig

design_id = -123
#design_id = 123
#design_id = 0xfedcba9876543210
visit0 = 456
date_obs = '2020-11-11'

d = makeConfig.makePfsDesign(design_id, np.array([1]), np.array([1]), np.array([1]), np.array([1]), np.array([1]))

c = StellaPfsConfig(d.pfsDesignId, visit0, d.raBoresight, d.decBoresight, d.fiberId, d.tract, d.patch, d.ra, d.dec, d.catId, d.objId,
                    d.targetType, d.fiberStatus, d.fiberMag, d.filterNames, d.pfiNominal, d.pfiNominal)

# This doesn't work - as class method in pfs.datamodel.PfsConfig doesn't use cls object to instantiate
#c = StellaPfsConfig.fromPfsDesign(d, visit0, d.pfiNominal)

dataId={'visit': visit0, 'pfsDesignId': design_id, 'dateObs': date_obs }

butler = dafPersist.Butler('.')
butler.put(c, 'pfsConfig', dataId)

