from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import json
from urllib.parse import parse_qs
from model_library.router.config import config_router
from model_library.router.infer import router as infer_router
from model_library.tools import log_api_complete

app = FastAPI(
    title="南山消防智能检测系统",
    description="提供消防相关的AI模型推理服务",
    version="1.0.0"
)

# 自定义Request类，支持重复读取body
class RequestWithCachedBody:
    def __init__(self, request: Request, body: bytes):
        self._request = request
        self._body = body
    
    def __getattr__(self, name):
        return getattr(self._request, name)
    
    async def body(self):
        return self._body

# API日志中间件
@app.middleware("http")
async def api_log_middleware(request: Request, call_next):
    """自动记录所有API请求和响应"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = str(request.url.path)
    
    # 获取请求参数
    request_params = {}
    original_body = b""
    
    try:
        # 1. 获取查询参数
        if request.query_params:
            request_params["query"] = dict(request.query_params)
        
        # 2. 获取请求体（POST请求）
        if method in ["POST", "PUT", "PATCH"]:
            # 读取原始请求体
            original_body = await request.body()
            
            if original_body:
                content_type = request.headers.get("content-type", "")
                
                if "application/json" in content_type:
                    # JSON数据
                    try:
                        request_params["body"] = json.loads(original_body.decode())
                    except:
                        request_params["body"] = {"raw_size": len(original_body)}
                
                elif "application/x-www-form-urlencoded" in content_type:
                    # Form数据
                    try:
                        form_data = parse_qs(original_body.decode())
                        # 转换为简单字典格式
                        request_params["form"] = {k: v[0] if len(v) == 1 else v for k, v in form_data.items()}
                    except:
                        request_params["form"] = {"parse_error": True}
                
                elif "multipart/form-data" in content_type:
                    # 文件上传，解析表单字段
                    try:
                        # 简单解析multipart数据（只获取字段名，不处理文件内容）
                        body_str = original_body.decode('utf-8', errors='ignore')
                        form_fields = {}
                        
                        # 查找表单字段
                        lines = body_str.split('\r\n')
                        current_field = None
                        for line in lines:
                            if 'name="' in line:
                                start = line.find('name="') + 6
                                end = line.find('"', start)
                                if end > start:
                                    current_field = line[start:end]
                            elif current_field and line.strip() and not line.startswith('-'):
                                form_fields[current_field] = line.strip()
                                current_field = None
                        
                        request_params["multipart"] = {
                            "fields": form_fields,
                            "size": len(original_body)
                        }
                    except:
                        request_params["multipart"] = {"size": len(original_body)}
                
                else:
                    # 其他类型，只记录大小
                    request_params["body"] = {"size": len(original_body), "content_type": content_type}
        
        # 重新构造request对象，因为body只能读取一次
        if original_body:
            request = RequestWithCachedBody(request, original_body)
    
    except Exception as e:
        request_params["parse_error"] = str(e)
    
    try:
        # 调用下一个处理器
        response = await call_next(request)
        duration = time.time() - start_time
        
        # 读取并记录响应内容
        response_data = {"status_code": response.status_code}
        
        try:
            # 读取响应体
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # 尝试解析响应内容
            if response_body:
                try:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        # JSON响应，解析并记录
                        response_json = json.loads(response_body.decode('utf-8'))
                        response_data["body"] = response_json
                    else:
                        # 非JSON响应，只记录大小
                        response_data["body_size"] = len(response_body)
                        response_data["content_type"] = content_type
                except json.JSONDecodeError:
                    # JSON解析失败，记录原始内容（截断）
                    body_text = response_body.decode('utf-8', errors='ignore')
                    response_data["body_text"] = body_text[:500] + "..." if len(body_text) > 500 else body_text
                except Exception as e:
                    response_data["body_parse_error"] = str(e)
            
            # 重新构造响应对象
            from fastapi.responses import Response as FastAPIResponse
            new_response = FastAPIResponse(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type")
            )
            
            # 记录成功的API调用
            log_api_complete(
                method=method,
                path=path,
                client_ip=client_ip,
                request_params=request_params,
                response_data=response_data,
                status_code=response.status_code,
                duration=duration
            )
            
            return new_response
            
        except Exception as response_error:
            # 响应处理失败，记录错误但仍然返回原响应
            response_data["response_read_error"] = str(response_error)
            
            log_api_complete(
                method=method,
                path=path,
                client_ip=client_ip,
                request_params=request_params,
                response_data=response_data,
                status_code=response.status_code,
                duration=duration
            )
            
            return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # 记录失败的API调用
        log_api_complete(
            method=method,
            path=path,
            client_ip=client_ip,
            request_params=request_params,
            status_code=500,
            duration=duration,
            error=str(e)
        )
        raise

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
    uvicorn.run(app, host="0.0.0.0", port=5122, access_log=False)  # 关闭默认访问日志，使用我们的中间件