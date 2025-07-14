from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from model_library.router.config import config_router
from model_library.router.infer import router as infer_router

app = FastAPI(
    title="南山消防智能检测系统",
    description="提供消防相关的AI模型推理服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

app.include_router(config_router, prefix="/ai_model")
app.include_router(infer_router,prefix="/ai_model")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5122)