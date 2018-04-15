import os
def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass