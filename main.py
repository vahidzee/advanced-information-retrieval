from src.mir import MIR
from src.api import API

mir = MIR()
api = API(mir)
mir.load_dataset()
api.run()

