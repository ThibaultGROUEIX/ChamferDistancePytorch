def get(opt):
    if opt.accelerated_chamfer:
        import dist_chamfer_idx as ext

        distChamfer = ext.chamferDist()
    else:
        import chamfer_python

        distChamfer = chamfer_python.distChamfer
    return distChamfer
