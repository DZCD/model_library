from fastapi import APIRouter
from model_library.tools.utils import Config
from fastapi.responses import JSONResponse

config_router = APIRouter(prefix="/config")


@config_router.get("/model")
def get_model_config():
    config  = Config()
    model_list = config.model_list
    return JSONResponse(content=model_list,status_code=200)
    

