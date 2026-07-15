"""生成更多中文长对话数据"""
import json

output_path = 'taiji_data/training_data/long_conversations.jsonl'

samples = []

conversations = [
    {
        'system': '你是态极，一个专业的AI助手。',
        'turns': [
            ('user', '什么是深度学习？它和传统机器学习有什么区别？'),
            ('assistant', '深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂模式。\n\n主要区别：\n\n1. 特征工程：\n- 传统ML：需要手动提取特征\n- 深度学习：自动学习特征\n\n2. 数据需求：\n- 传统ML：小数据集也能工作\n- 深度学习：需要大量数据\n\n3. 计算资源：\n- 传统ML：CPU即可\n- 深度学习：通常需要GPU\n\n4. 模型复杂度：\n- 传统ML：模型简单，可解释性强\n- 深度学习：模型复杂，黑箱\n\n常见深度学习模型：\n- CNN：图像处理\n- RNN/LSTM：序列数据\n- Transformer：NLP和多模态\n- GAN：生成模型'),
            ('user', '能详细解释一下CNN吗？'),
            ('assistant', 'CNN（卷积神经网络）是专门处理网格结构数据（如图像）的神经网络。\n\n核心组件：\n\n1. 卷积层（Convolutional Layer）：\n- 使用滤波器（kernel）提取局部特征\n- 参数共享减少计算量\n- 输出：特征图（feature map）\n\n2. 池化层（Pooling Layer）：\n- 降采样，减少参数\n- 最大池化、平均池化\n- 增强平移不变性\n\n3. 全连接层（Fully Connected）：\n- 将特征映射到输出\n- 用于分类或回归\n\n典型架构：\n输入 -> [卷积 -> 激活 -> 池化] x N -> 全连接 -> 输出\n\n经典模型：\n- LeNet-5：手写数字识别\n- AlexNet：ImageNet冠军\n- VGG：深层网络\n- ResNet：残差连接\n- EfficientNet：高效架构\n\n应用场景：\n- 图像分类\n- 目标检测\n- 图像分割\n- 人脸识别'),
        ]
    },
    {
        'system': '你是态极，一个有帮助的编程助手。',
        'turns': [
            ('user', '什么是微服务架构？和单体架构有什么区别？'),
            ('assistant', '微服务架构是将应用拆分为多个小型、独立服务的架构风格。\n\n对比：\n\n单体架构：\n- 所有功能在一个应用中\n- 共享数据库\n- 部署简单但扩展困难\n- 技术栈统一\n\n微服务架构：\n- 每个服务独立部署\n- 每个服务有自己的数据库\n- 可独立扩展\n- 技术栈可不同\n\n微服务优势：\n1. 独立部署：修改一个服务不影响其他\n2. 技术灵活：不同服务可用不同语言\n3. 弹性扩展：按需扩展热点服务\n4. 故障隔离：一个服务挂了不影响整体\n\n微服务挑战：\n1. 网络复杂：服务间通信\n2. 数据一致性：分布式事务\n3. 运维复杂：需要容器编排\n4. 调试困难：分布式追踪\n\n常用技术：\n- 服务发现：Consul, Eureka\n- API网关：Kong, Nginx\n- 容器编排：Docker, Kubernetes\n- 消息队列：RabbitMQ, Kafka'),
            ('user', '如何设计一个好的微服务？'),
            ('assistant', '设计好的微服务需要遵循以下原则：\n\n1. 单一职责：\n- 每个服务只做一件事\n- 职责清晰，边界明确\n- 例：用户服务、订单服务、支付服务\n\n2. 服务自治：\n- 独立开发、测试、部署\n- 拥有自己的数据存储\n- 不依赖其他服务的内部实现\n\n3. 接口设计：\n- RESTful API 或 gRPC\n- 版本管理\n- 文档完善\n\n4. 数据管理：\n- 每个服务独立数据库\n- 避免直接访问其他服务的数据库\n- 使用事件驱动同步数据\n\n5. 通信方式：\n- 同步：HTTP/gRPC（实时响应）\n- 异步：消息队列（解耦）\n\n6. 容错设计：\n- 超时控制\n- 熔断器（Circuit Breaker）\n- 重试机制\n- 降级策略\n\n7. 监控和追踪：\n- 日志聚合\n- 指标监控\n- 分布式追踪\n\n最佳实践：\n- 从单体开始，逐步拆分\n- 按业务领域划分服务\n- 自动化CI/CD\n- 容器化部署'),
        ]
    },
    {
        'system': '你是态极，一个专业的AI助手。',
        'turns': [
            ('user', '什么是Transformer架构？为什么它这么重要？'),
            ('assistant', 'Transformer是2017年Google提出的神经网络架构，彻底改变了NLP领域。\n\n核心创新：\n\n1. 自注意力机制（Self-Attention）：\n- 计算序列中每个位置与其他位置的关联\n- 捕捉长距离依赖\n- 并行计算，效率高\n\n2. 多头注意力：\n- 同时关注不同子空间\n- 增强模型表达能力\n\n3. 位置编码：\n- 注入序列位置信息\n- 因为注意力机制本身没有顺序概念\n\n4. 编码器-解码器结构：\n- 编码器：理解输入\n- 解码器：生成输出\n\n为什么重要：\n\n1. 并行化：比RNN快很多\n2. 长距离依赖：能捕捉远距离关系\n3. 可扩展性：参数量越大，能力越强\n4. 通用性：NLP、CV、多模态都能用\n\n衍生模型：\n- BERT：双向编码器\n- GPT：自回归解码器\n- ViT：视觉Transformer\n- CLIP：多模态\n- LLaMA：开源大模型\n\n现代AI革命的基础架构！'),
            ('user', '能详细解释一下注意力机制吗？'),
            ('assistant', '注意力机制是Transformer的核心，我来详细解释：\n\n基本概念：\n注意力机制让模型在处理每个位置时，能"关注"到输入序列的所有位置。\n\n数学公式：\nAttention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V\n\n其中：\n- Q（Query）：查询向量，当前位置\n- K（Key）：键向量，所有位置\n- V（Value）：值向量，所有位置\n- d_k：Key的维度\n\n计算过程：\n1. 计算Q和K的点积：得到注意力分数\n2. 缩放：除以sqrt(d_k)，防止梯度消失\n3. Softmax：归一化为概率分布\n4. 加权求和：用注意力分数对V加权\n\n多头注意力：\n- 将Q, K, V分成多个头\n- 每个头独立计算注意力\n- 拼接结果后线性变换\n- 好处：关注不同子空间的信息\n\n示例：\n句子："猫坐在垫子上"\n当处理"坐"时：\n- 高注意力：猫（主语）、垫子（位置）\n- 低注意力：在、上\n\n可视化：\n```\n       猫  坐  在  垫子  上\n猫   [0.1 0.3 0.1 0.4  0.1]\n坐   [0.4 0.1 0.1 0.3  0.1]\n在   [0.1 0.2 0.1 0.5  0.1]\n垫子 [0.3 0.2 0.1 0.3  0.1]\n上   [0.1 0.1 0.2 0.5  0.1]\n```\n\n这就是为什么Transformer能理解语言的含义！'),
        ]
    },
    {
        'system': '你是态极，一个专业的编程助手。',
        'turns': [
            ('user', '什么是API？如何设计一个好的API？'),
            ('assistant', 'API（应用程序编程接口）是软件组件之间交互的契约。\n\nAPI类型：\n1. REST API：基于HTTP，使用JSON\n2. GraphQL：灵活查询，按需获取\n3. gRPC：高性能，基于Protocol Buffers\n4. WebSocket：实时双向通信\n\nREST API设计原则：\n\n1. 资源导向：\n- 用名词表示资源：/users, /orders\n- 用HTTP方法表示操作：GET, POST, PUT, DELETE\n\n2. URL设计：\n- 层级关系：/users/123/orders\n- 查询参数：/users?role=admin&status=active\n- 避免动词：用POST /users 而不是 /createUser\n\n3. 状态码：\n- 200：成功\n- 201：创建成功\n- 400：客户端错误\n- 401：未认证\n- 403：无权限\n- 404：资源不存在\n- 500：服务器错误\n\n4. 版本管理：\n- URL版本：/api/v1/users\n- Header版本：Accept: application/vnd.api.v1+json\n\n5. 认证授权：\n- JWT Token\n- OAuth 2.0\n- API Key\n\n6. 错误处理：\n```json\n{\n  "error": {\n    "code": "INVALID_INPUT",\n    "message": "邮箱格式不正确",\n    "details": {"field": "email"}\n  }\n}\n```\n\n7. 分页：\n```\nGET /users?page=1&limit=20\nResponse: {\n  "data": [...],\n  "pagination": {\n    "page": 1,\n    "limit": 20,\n    "total": 100\n  }\n}\n```\n\n8. 文档：\n- OpenAPI/Swagger\n- 自动生成文档\n- 提供示例'),
            ('user', 'REST和GraphQL哪个更好？'),
            ('assistant', 'REST和GraphQL各有优劣，选择取决于场景：\n\nREST优势：\n1. 简单直观：标准HTTP方法\n2. 缓存友好：GET请求可缓存\n3. 生态成熟：工具和最佳实践丰富\n4. 学习曲线低：容易上手\n\nREST劣势：\n1. 过度获取：返回所有字段\n2. 不足获取：需要多次请求\n3. 版本管理：URL或Header版本\n\nGraphQL优势：\1. 灵活查询：客户端决定返回什么字段\n2. 单次请求：获取多个资源\n3. 强类型：Schema定义数据结构\n4. 无版本：通过字段演进\n\nGraphQL劣势：\n1. 复杂性：学习曲线陡峭\n2. 缓存困难：POST请求不缓存\n3. N+1问题：需要DataLoader优化\n4. 安全风险：复杂查询可能耗尽资源\n\n选择建议：\n\n用REST：\n- 简单CRUD应用\n- 需要HTTP缓存\n- 团队熟悉REST\n- 公开API\n\n用GraphQL：\n- 复杂数据关系\n- 多端（Web、Mobile）不同需求\n- 需要灵活查询\n- 内部API\n\n混合使用：\n- REST：简单操作、文件上传\n- GraphQL：复杂查询、数据聚合\n\n实际案例：\n- GitHub：同时提供REST和GraphQL\n- Shopify：主推GraphQL\n- 大多数公司：REST为主'),
        ]
    },
    {
        'system': '你是态极，一个专业的AI助手。',
        'turns': [
            ('user', '什么是数据库索引？如何优化数据库查询？'),
            ('assistant', '数据库索引是帮助快速查找数据的数据结构，类似书的目录。\n\n索引类型：\n\n1. B+树索引（最常用）：\n- 适合范围查询\n- 支持排序\n- 大多数数据库默认\n\n2. 哈希索引：\n- 精确匹配查询\n- 不支持范围查询\n- 速度极快\n\n3. 全文索引：\n- 文本搜索\n- 支持模糊匹配\n\n4. 空间索引：\n- 地理位置查询\n\n索引优缺点：\n\n优点：\n- 加速查询（10-1000倍）\n- 减少磁盘IO\n\n缺点：\n- 占用存储空间\n- 降低写入速度（需要维护索引）\n- 可能导致索引失效\n\n索引设计原则：\n\n1. 选择性高的列：\n- 性别（男/女）：选择性低，不适合\n- 邮箱：选择性高，适合\n\n2. 查询条件列：\n- WHERE子句的列\n- JOIN条件的列\n- ORDER BY的列\n\n3. 复合索引：\n- 最左前缀原则\n- (a, b, c) 支持 a, a+b, a+b+c\n\n查询优化技巧：\n\n1. 使用EXPLAIN分析：\n```sql\nEXPLAIN SELECT * FROM users WHERE email = \'test@example.com\';\n```\n\n2. 避免索引失效：\n- 不要在索引列上使用函数\n- 避免隐式类型转换\n- 避免前导通配符：\'%abc\'\n\n3. 查询优化：\n- 只查需要的列：SELECT id, name 而不是 SELECT *\n- 使用LIMIT限制结果\n- 避免子查询，用JOIN\n\n4. 表设计：\n- 合适的数据类型\n- 范式化与反范式化平衡\n- 分区表\n\n监控工具：\n- 慢查询日志\n- 索引使用统计\n- 性能分析器'),
            ('user', '什么时候不应该使用索引？'),
            ('assistant', '索引不是万能的，以下情况应该避免使用：\n\n1. 小表：\n- 数据量少（几百行）\n- 全表扫描比索引更快\n- 维护索引的开销不值得\n\n2. 频繁更新的列：\n- 每次更新都要维护索引\n- 写入性能下降\n- 权衡读写比例\n\n3. 低选择性列：\n- 性别：男/女（只有2个值）\n- 状态：启用/禁用\n- 索引效果差\n\n4. 大字段：\n- TEXT、BLOB类型\n- 索引占用空间大\n- 使用前缀索引或全文索引\n\n5. 读少写多的表：\n- 日志表\n- 消息表\n- 索引维护成本高\n\n6. 重复值多的列：\n- 索引选择性差\n- 扫描索引不如全表扫描\n\n索引失效场景：\n\n1. 使用函数：\n```sql\n-- 索引失效\nWHERE YEAR(create_time) = 2024\n-- 优化\nWHERE create_time >= \'2024-01-01\' AND create_time < \'2025-01-01\'\n```\n\n2. 隐式转换：\n```sql\n-- phone是varchar，传入数字\nWHERE phone = 13800138000  -- 失效\nWHERE phone = \'13800138000\'  -- 有效\n```\n\n3. 前导通配符：\n```sql\nWHERE name LIKE \'%abc\'  -- 失效\nWHERE name LIKE \'abc%\'  -- 有效\n```\n\n4. OR条件：\n```sql\n-- 如果or的列没有都建索引\nWHERE a = 1 OR b = 2  -- 可能失效\n```\n\n最佳实践：\n- 定期分析索引使用情况\n- 删除未使用的索引\n- 监控慢查询\n- 测试索引效果'),
        ]
    },
]

for conv in conversations:
    messages = [{'role': 'system', 'content': conv['system']}]
    for role, content in conv['turns']:
        messages.append({'role': role, 'content': content})
    samples.append({'messages': messages})

with open(output_path, 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

total_turns = sum(len(c['turns']) for c in conversations)
print(f'Generated {len(samples)} conversations ({total_turns} turns) -> {output_path}')
