# Knowledge Retriever

# Env Prepare:
```bash
# 激活虚拟环境
conda activate xxx / source ${VIRTUAL_ENV_PATH}/bin/activate
# 更新虚拟环境中的pip包
pip install --upgrade pip
# 在虚拟环境中安装poetry
pip install poetry
# 通过poetry进行依赖包安装
poetry install
```

# Start Spec:
```bash
cp .env .env

# 修改.env文件中的配置
vim .env

# 启动服务，注意保持.env在启动的当前目录下
python -m KnowledgeRetriever

# 等待服务启动完成，进入swagger页面：http://localhost:8080/docs
```

# Project Structure Spec
```bash
.
├── README.md                   (项目环境说明)
├── pyproject.toml              (基于poetry的项目依赖管理)
├── KnowledgeRetriever          (项目代码)
    ├── handlers                (业务处理包)
        ├── dataplane.py        (***实现具体业务处理逻辑, 类比service层, 例如调用报告问答服务...)
    ├── rest                    (REST服务包)
        ├── app.py              (Fastapi应用程序, 主要负责:定义REST端点)
        ├── endpoints.py        (***REST端点的实现, 主要负责端点输入/输出处理, 具体实现通过dataplane.py模块完成)
        ├── errors.py           (定义Fastapi应用程序自动异常处理)
        ├── logging.py          (封装logging模块, 用于统一日志管理)
        ├── requests.py         (封装Fastapi的Request, 用于统一请求处理，用于编解码, 暂时用不到)
        ├── responses.py        (封装Fastapi的Response, 用于统一相应处理，用于编解码, 暂时用不到)
        ├── server.py           (封装Uvicorn, 主要负责启动、停止Uvicorn服务器)
    ├── schemas                 (数据模型包)
        ├── input_schemas.py    (***请求输入数据模型)
        ├── output_schemas.py   (***响应输出数据模型)
    ├── utils                   (文档生成项目全局工具包)
        ├── logging.py          (日志工具模块)
    ├── errors.py               (***文档生成项目全局异常模块, 用于端点实现接口中使用的自定义异常)
    ├── settings.py             (***文档生成项目配置)
    ├── server.py               (文档生成项目启动、停止)
    ├── version.py              (文档生成项目迭代版本)
    ├── __main__.py             (文档生成项目启动入口点)
```

### neo4j
`docker run -it -d -p 7474:7474 -p 7687:7687 -v .\neo4j\data:/data --name neo4j neo4j:5.18.0`

### Tips
部分核心代码文件开发时使用了相对路径，在使用时可能会有问题，请仔细甄别。