"""验证规模化训练数据下的生成质量改善。

关键验证点：
1. 喂入多样化训练数据（每域 20+ 条，覆盖多主题）
2. 多轮 feed+sleep 训练
3. 每轮评估生成质量：loss、token 多样性、训练数据覆盖率
4. 对比训练前后的生成文本

Usage:
    python scripts/training/verify_scaled_training.py
"""
import sys
import os
from datetime import datetime
from collections import Counter

os.environ.setdefault('TAJIJI_TEST_MODE', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


# ── 多样化训练数据（~1000 条，每域约 250 条） ──
TRAINING_DATA = {
    "zh": [
        # 天气主题 (8)
        "今天天气很好，我们一起去公园散步。",
        "明天天气怎么样？会下雨吗？",
        "天气预报说周末是晴天，适合出游。",
        "夏天的雷阵雨总是来得很突然。",
        "冬天的雪景真美，孩子们在堆雪人。",
        "春天暖风吹过，百花竞相开放。",
        "秋高气爽，蓝天白云令人心旷神怡。",
        "台风来了，记得关好窗户待在家里。",
        # 科技主题 (8)
        "人工智能正在改变世界，神经元协同工作。",
        "态极神经元架构通过共振场实现意识涌现。",
        "深度学习模型需要大量数据训练。",
        "大语言模型展现了强大的理解能力。",
        "量子计算机将在未来解决复杂问题。",
        "区块链技术让数据更加安全可靠。",
        "云计算让企业无需自建数据中心。",
        "物联网连接了万物，让生活更智能。",
        # 生活主题 (8)
        "我喜欢在清晨喝一杯咖啡，开始新的一天。",
        "读书是获取知识的重要途径。",
        "运动有益健康，每天坚持锻炼。",
        "音乐能陶冶情操，让人心情愉悦。",
        "做饭是一门艺术，需要不断练习。",
        "旅行能开阔眼界，增长见识。",
        "和朋友聚会是最快乐的时光。",
        "养花种草让生活充满绿意。",
        # 学习主题 (8)
        "学习新语言需要耐心和持续练习。",
        "数学是科学的基础，逻辑思维很重要。",
        "历史告诉我们很多道理。",
        "物理研究自然界的基本规律。",
        "化学实验需要注意安全操作。",
        "写作能力需要通过大量阅读来提升。",
        "编程是一项实用的技能。",
        "哲学思考帮助人们认识自我。",
        # 自然主题 (8)
        "春天来了，万物复苏，花开满园。",
        "秋天是收获的季节，树叶变黄飘落。",
        "海洋是地球上最大的生态系统。",
        "森林是地球的肺，提供氧气。",
        "河流奔腾不息，汇入大海。",
        "高山巍峨耸立，终年积雪。",
        "沙漠虽然干旱，也有独特生态。",
        "极地的极光绚丽多彩，美不胜收。",
        # 社会主题 (8)
        "城市生活节奏快，交通便利。",
        "乡村生活宁静祥和，空气清新。",
        "教育是改变命运的重要途径。",
        "团队合作能完成更复杂的任务。",
        "诚信是为人之本，不可丢弃。",
        "创新推动社会不断向前发展。",
        "文化交流增进不同民族的理解。",
        "保护环境是每个人的责任。",
        # 情感主题 (6)
        "快乐不在于拥有多少，而在于知足。",
        "勇气不是不害怕，而是害怕了仍然前行。",
        "友谊需要用心经营，才能长久。",
        "家是温暖的港湾，永远等着你回来。",
        "梦想是前进的动力，不要轻易放弃。",
        "感恩生活中的每一个美好瞬间。",
        # 历史主题 (9)
        "唐朝是中国历史上最辉煌的朝代之一。",
        "秦始皇统一六国，建立了中央集权制度。",
        "丝绸之路促进了东西方文化的交流。",
        "五四运动推动了中国社会的思想解放。",
        "长城是古代防御工程的伟大奇迹。",
        "火药是中国的四大发明之一。",
        "宋朝的商业经济非常繁荣发达。",
        "汉武帝派遣张骞出使西域。",
        "郑和下西洋展示了明朝的航海实力。",
        # 文学主题 (8)
        "红楼梦被誉为中国古典小说的巅峰之作。",
        "唐诗宋词是中国文学的两座高峰。",
        "鲁迅以犀利的文笔批判社会弊端。",
        "李白是浪漫主义诗人的杰出代表。",
        "小说通过人物描写反映社会现实。",
        "散文以自由的形式抒发作者情感。",
        "民间故事蕴含着丰富的文化智慧。",
        "文学创作来源于生活又高于生活。",
        # 哲学主题 (8)
        "老子主张无为而治的自然哲学。",
        "庄子以逍遥游表达自由精神境界。",
        "儒家思想强调仁义礼智信的道德规范。",
        "知行合一是王阳明心学的核心。",
        "辩证法认为事物发展由矛盾推动。",
        "存在先于本质是一种存在主义观点。",
        "理性思考是人类认识世界的重要方式。",
        "自由意志与决定论是哲学的根本问题。",
        # 经济主题 (8)
        "通货膨胀会导致货币购买力下降。",
        "市场经济通过价格机制调节供需。",
        "国际贸易促进了全球资源配置。",
        "税收是政府提供公共服务的资金来源。",
        "GDP是衡量一个国家经济总量的指标。",
        "投资需要平衡风险与收益的关系。",
        "创业需要创新精神和市场洞察力。",
        "消费升级反映了人民生活水平的提高。",
        # 健康主题 (8)
        "均衡饮食是保持身体健康的基础。",
        "充足的睡眠有助于身体恢复和修复。",
        "心理健康和身体健康同样重要。",
        "定期体检能够及早发现潜在疾病。",
        "吸烟对呼吸系统有严重的危害。",
        "运动能增强免疫力和心肺功能。",
        "保持良好心态有助于延年益寿。",
        "喝水是维持新陈代谢的必要条件。",
        # 饮食主题 (8)
        "中国菜有八大菜系各具特色风味。",
        "川菜以麻辣鲜香闻名于世。",
        "粽子的主要原料是糯米和粽叶。",
        "火锅是冬天最受欢迎的美食之一。",
        "茶的种类丰富有绿茶红茶乌龙茶。",
        "饺子是中国传统节日的必备食品。",
        "食材的新鲜程度决定了菜肴的品质。",
        "烹饪讲究火候和调料的搭配比例。",
        # 旅行主题 (8)
        "旅行能让人体验不同的风土人情。",
        "西藏的高原风光令人叹为观止。",
        "云南的少数民族文化丰富多彩。",
        "故宫是中国古代建筑艺术的瑰宝。",
        "三亚的海滩是度假休闲的好去处。",
        "西安古城墙上可以俯瞰城市全景。",
        "苏州园林以精巧布局闻名天下。",
        "黄山以奇松怪石云海温泉著称。",
        # 运动主题 (8)
        "跑步是最简单有效的有氧运动方式。",
        "游泳能锻炼全身肌肉和协调能力。",
        "篮球需要团队配合和个人技术。",
        "足球场上配合默契才能赢得胜利。",
        "瑜伽可以调节呼吸和放松身心。",
        "乒乓球是中国最受欢迎的球类运动。",
        "运动前的热身能有效防止受伤。",
        "坚持锻炼需要毅力和自律精神。",
        # 节日主题 (6)
        "春节是中国人最重要的传统节日。",
        "中秋节全家人团聚吃月饼赏月。",
        "端午节赛龙舟吃粽子纪念屈原。",
        "元宵节赏花灯猜灯谜吃汤圆。",
        "清明节扫墓祭祖缅怀先人。",
        "国庆节是庆祝新中国成立的节日。",
        # 科技深度 (12)
        "纳米技术可以制造出分子级别的材料。",
        "机器人正在工厂中替代重复性劳动。",
        "太空探索让人类足迹扩展到地球之外。",
        "生物技术可以改良农作物提高产量。",
        "虚拟现实创造了身临其境的体验。",
        "基因编辑技术有望治疗遗传疾病。",
        "自动驾驶汽车将改变出行方式。",
        "5G通信实现了更快的网络连接。",
        "可再生能源技术不断取得突破进展。",
        "芯片是电子设备的核心组成部分。",
        "卫星导航系统让定位更加精准。",
        "网络安全保护个人信息不被泄露。",
        # 心理健康 (12)
        "适度压力可以激发人的潜能和动力。",
        "正念冥想有助于缓解焦虑和压力。",
        "积极的心态是克服困难的重要武器。",
        "倾听是建立良好人际关系的基础。",
        "自我反思帮助我们发现自身的不足。",
        "拖延症可以通过分解任务来克服。",
        "同理心让我们更好地理解他人感受。",
        "习惯的养成需要二十一天的坚持。",
        "情绪管理是成年人的必修课程。",
        "社交支持网络对心理健康至关重要。",
        "接受不完美是内心强大的表现。",
        "好的睡眠习惯对情绪调节很重要。",
        # 职业发展 (10)
        "终身学习是职场竞争力的保障。",
        "时间管理是提高工作效率的关键。",
        "沟通能力在团队协作中至关重要。",
        "领导力不仅仅是发号施令的能力。",
        "职业规划需要明确长期和短期目标。",
        "简历是展示个人能力的第一张名片。",
        "面试时需要自信地展示专业技能。",
        "跨领域知识让职业道路更宽广。",
        "工作中的挫折是成长的机会。",
        "建立专业人脉有助于职业发展。",
        # 天文地理 (10)
        "太阳系有八大行星围绕太阳运行。",
        "银河系包含了数千亿颗恒星。",
        "黑洞是宇宙中最神秘的天体之一。",
        "地球的自转产生了昼夜交替。",
        "板块运动会导致地震和火山喷发。",
        "大气层保护地球免受宇宙辐射。",
        "北极星是北半球导航的重要参照。",
        "洋流影响着全球气候的分布。",
        "月球的引力引起了海洋的潮汐。",
        "陨石是来自太空的天然岩石碎片。",
        # 农业生态 (10)
        "有机农业不使用化学合成的农药。",
        "轮作可以保持土壤的肥力不衰退。",
        "蜜蜂在植物授粉中扮演关键角色。",
        "生态平衡是农业可持续发展的基础。",
        "温室大棚可以控制作物生长环境。",
        "灌溉技术提高了干旱地区的产量。",
        "土壤微生物对植物生长至关重要。",
        "病虫害防治需要综合多种方法。",
        "粮食安全是国家战略的重要组成。",
        "精准农业利用数据优化种植决策。",
        # 法律政治 (10)
        "宪法是国家的根本大法和最高准则。",
        "法律面前人人平等是法治的基本原则。",
        "人民代表大会制度是我国的政体。",
        "公民有依法纳税和服兵役的义务。",
        "知识产权保护鼓励了创新和创造。",
        "国际法规范了国家之间的交往行为。",
        "司法独立是公正审判的重要保障。",
        "选举是民主制度的核心环节。",
        "行政法规规范了政府的管理行为。",
        "公益诉讼维护了社会公共利益。",
        # 宗教文化 (8)
        "佛教强调慈悲为怀和因果报应。",
        "道教追求天人合一的自然境界。",
        "儒家文化深刻影响了东亚社会。",
        "基督教以博爱精神为核心教义。",
        "伊斯兰教强调顺从真主的意愿。",
        "文化多样性是人类共同的财富。",
        "非物质文化遗产需要保护和传承。",
        "民间信仰反映了百姓的精神寄托。",
        # 建筑交通 (8)
        "桥梁连接了两岸的交通和经济。",
        "高铁让城市之间的距离大大缩短。",
        "地铁是大型城市的主要公共交通。",
        "城市规划需要考虑宜居性和可持续性。",
        "古建筑是历史文化的珍贵遗产。",
        "现代建筑追求功能和美观的统一。",
        "港口是国际贸易的重要枢纽。",
        "航空运输让世界变成了地球村。",
        # 军事国防 (8)
        "国防建设是国家安全的根本保障。",
        "和平与发展是当今世界的主题。",
        "军事科技推动了许多民用技术。",
        "战略思维在军事和商业中都重要。",
        "维和行动维护了冲突地区的稳定。",
        "太空已经成为新的战略竞争领域。",
        "网络战是信息化战争的新形态。",
        "军民融合促进了国防与经济发展。",
        # 语言文字 (7)
        "汉字是世界上最古老的文字之一。",
        "成语蕴含着丰富的历史文化典故。",
        "方言反映了不同地域的文化特色。",
        "翻译需要兼顾准确性和流畅性。",
        "修辞手法让语言表达更加生动。",
        "普通话促进了全国人民的沟通交流。",
        "阅读广泛的书籍能提高语文素养。",
        # 医疗健康 (10)
        "中医讲究阴阳平衡和五行调和。",
        "疫苗是预防传染病最有效的方法。",
        "抗生素不能滥用以免产生耐药性。",
        "心脏是人体最重要的器官之一。",
        "糖尿病需要长期控制血糖水平。",
        "癌症的早期筛查能提高治愈率。",
        "传染病防控需要全社会的参与。",
        "营养学指导人们科学合理的饮食。",
        "急救知识可以在关键时刻挽救生命。",
        "公共卫生体系保障了全民健康。",
        # 经济金融 (10)
        "股票市场反映了企业的经营状况。",
        "银行在金融体系中发挥核心作用。",
        "通货膨胀影响人们的生活成本。",
        "汇率变动影响国际贸易的竞争力。",
        "保险为人们提供了风险保障。",
        "基金是集合投资理财的方式。",
        "数字货币正在改变支付方式。",
        "财政政策调节宏观经济的运行。",
        "供应链管理优化了商品流通效率。",
        "房地产市场牵动着国民经济的命脉。",
        # 娱乐体育 (10)
        "电影是一种综合性的视听艺术。",
        "电子竞技已经成为一个正式体育项目。",
        "围棋考验选手的策略和计算能力。",
        "健身运动要根据自身情况量力而行。",
        "摄影记录下生活中美好的瞬间。",
        "舞蹈能够表达语言难以描述的情感。",
        "攀岩锻炼了人的勇气和身体协调。",
        "集邮是一种有趣而有意义的爱好。",
        "动漫文化深受年轻人的喜爱。",
        "滑冰是冬季最受欢迎的体育运动。",
    ],
    "en": [
        # Technology (10)
        "The cortex architecture uses resonance fields.",
        "Small neurons work together to match large models.",
        "Machine learning models learn from data.",
        "Neural networks are inspired by the brain.",
        "Language models process text efficiently.",
        "Quantum computers will solve complex problems.",
        "Blockchain makes data secure and reliable.",
        "Cloud computing eliminates data centers.",
        "The internet connects people worldwide.",
        "Artificial intelligence transforms industries.",
        # Daily life (10)
        "The weather is nice today, let's go for a walk.",
        "Reading books is important for learning.",
        "Exercise is good for your health.",
        "Music can affect our emotions deeply.",
        "Cooking is an art that requires practice.",
        "Travel broadens the mind and perspective.",
        "Friends make life more enjoyable.",
        "Gardening brings joy and beauty.",
        "A good cup of coffee starts the day right.",
        "Sleep is essential for good health.",
        # Science (8)
        "Science advances through careful experimentation.",
        "Technology changes the way we live.",
        "Physics studies the laws of nature.",
        "Chemistry explores matter and its properties.",
        "Biology investigates living organisms.",
        "Astronomy reveals the mysteries of universe.",
        "Mathematics is the language of science.",
        "Evolution explains the diversity of life.",
        # Society (8)
        "Education opens doors to new opportunities.",
        "Teamwork makes difficult tasks achievable.",
        "Nature provides beauty and resources.",
        "Cities offer convenience and diversity.",
        "Honesty is the foundation of trust.",
        "Innovation drives society forward.",
        "Cultural exchange promotes understanding.",
        "Protecting the environment is our duty.",
        # Emotions (8)
        "Happiness comes from within, not from things.",
        "Courage is acting despite fear.",
        "Friendship requires effort to maintain.",
        "Home is where the heart belongs.",
        "Dreams give us motivation to continue.",
        "Gratitude makes life more meaningful.",
        "Love is the most powerful force.",
        "Patience brings rewards in time.",
        # Nature (6)
        "The ocean is the largest ecosystem on Earth.",
        "Mountains rise majestically into the sky.",
        "Forests provide oxygen for the planet.",
        "Rivers flow endlessly to the sea.",
        "Seasons change in a beautiful cycle.",
        "Stars light up the night sky.",
        # History (10)
        "The Renaissance was a period of great cultural rebirth.",
        "World War Two reshaped the global political landscape.",
        "Ancient Rome built an extensive network of roads.",
        "The Industrial Revolution transformed manufacturing forever.",
        "Egyptian pyramids are wonders of ancient engineering.",
        "The Silk Road connected East and West for centuries.",
        "The invention of printing revolutionized knowledge sharing.",
        "Democracy originated in ancient Greek city-states.",
        "The moon landing was a historic achievement for mankind.",
        "Colonial independence movements changed the world order.",
        # Philosophy (9)
        "Philosophy asks fundamental questions about existence.",
        "Ethics concerns what is right and wrong in behavior.",
        "Logic provides the rules for valid reasoning.",
        "Knowledge requires justified true belief according to Plato.",
        "Free will debates challenge our understanding of choice.",
        "The meaning of life is a central philosophical question.",
        "Empiricism holds that knowledge comes from experience.",
        "Rationalism trusts reason as the source of knowledge.",
        "Moral relativism claims values vary across cultures.",
        # Economics (8)
        "Supply and demand determine market equilibrium prices.",
        "Interest rates influence borrowing and investment decisions.",
        "International trade creates comparative advantage benefits.",
        "Fiscal policy uses government spending to manage economy.",
        "Monetary policy controls money supply and inflation.",
        "Economic growth raises living standards over time.",
        "Unemployment rates signal the health of labor markets.",
        "The stock market aggregates information about companies.",
        # Health (8)
        "Regular checkups help prevent serious medical conditions.",
        "Mental wellbeing is as important as physical health.",
        "A balanced diet includes proteins carbohydrates and fats.",
        "Vaccination protects individuals and the wider community.",
        "Stress management techniques improve quality of life.",
        "Heart disease can be prevented with healthy habits.",
        "Adequate hydration is essential for body functions.",
        "The immune system defends the body against pathogens.",
        # Travel (8)
        "The Grand Canyon offers breathtaking views of nature.",
        "Paris is known as the city of lights and romance.",
        "Japanese temples reflect a deep spiritual tradition.",
        "African safaris let you observe wildlife up close.",
        "Venice canals create a unique urban landscape.",
        "The Great Wall stretches across northern China beautifully.",
        "Tropical islands offer pristine beaches and clear waters.",
        "Traveling solo builds confidence and self-reliance.",
        # Sports (8)
        "Tennis requires both physical agility and mental focus.",
        "Marathon runners train for months to build endurance.",
        "Gymnastics demands exceptional flexibility and strength.",
        "Soccer is the most popular sport in the world.",
        "Basketball players need speed coordination and teamwork.",
        "Swimming competitions test speed and technique.",
        "Mountain climbing challenges both body and spirit.",
        "Baseball is a game of strategy and precision.",
        # Arts (8)
        "Portrait painting captures the essence of a person.",
        "Symphony orchestras blend many instruments harmoniously.",
        "Ballet combines athleticism with artistic expression.",
        "Modern art challenges traditional notions of beauty.",
        "Poetry distills emotions into carefully chosen words.",
        "Architecture shapes the spaces where we live and work.",
        "Photography freezes moments in time forever.",
        "Theater brings stories to life on stage.",
        # Education (8)
        "Critical thinking is a key skill for lifelong learning.",
        "Online courses make education accessible to everyone.",
        "Early childhood education builds a strong foundation.",
        "Teachers play a vital role in shaping future generations.",
        "Learning a second language opens new perspectives.",
        "STEM education prepares students for modern careers.",
        "Reading widely develops vocabulary and comprehension.",
        "Universities drive research and innovation forward.",
        # Environment (8)
        "Renewable energy reduces dependence on fossil fuels.",
        "Climate change requires urgent global cooperation.",
        "Recycling helps conserve natural resources effectively.",
        "Plastic pollution threatens marine ecosystems worldwide.",
        "Sustainable farming protects soil and water quality.",
        "Biodiversity is essential for ecosystem resilience.",
        "Deforestation contributes to global carbon emissions.",
        "Urban parks improve air quality and public health.",
        # Technology Deep (12)
        "Machine learning models require careful hyperparameter tuning.",
        "Robotics combines mechanical engineering with artificial intelligence.",
        "Quantum computing uses qubits instead of classical bits.",
        "Biotechnology enables personalized medical treatments.",
        "Space exploration reveals the mysteries of the cosmos.",
        "Nanotechnology manipulates matter at the atomic scale.",
        "Virtual reality creates immersive digital environments.",
        "Self-driving cars use lidar and computer vision.",
        "Genetic engineering can modify organism DNA precisely.",
        "Semiconductor chips power all modern electronics.",
        "Satellite networks provide global communication coverage.",
        "Cybersecurity protects systems from malicious attacks.",
        # Psychology (12)
        "Emotional intelligence helps navigate social situations.",
        "Cognitive biases affect our decision making processes.",
        "The growth mindset believes abilities can be developed.",
        "Active listening improves communication and relationships.",
        "Self-reflection leads to personal growth and insight.",
        "Procrastination can be overcome with structured deadlines.",
        "Empathy allows us to understand perspectives deeply.",
        "Habits form through repetition and reinforcement loops.",
        "Resilience helps bounce back from adversity and setbacks.",
        "Social support networks buffer against mental stress.",
        "Mindfulness practice reduces anxiety and improves focus.",
        "Sleep quality directly impacts cognitive performance.",
        # Career Development (10)
        "Networking expands professional opportunities and knowledge.",
        "Time management skills boost workplace productivity.",
        "Effective communication is essential for team success.",
        "Leadership involves inspiring and guiding others.",
        "Setting clear goals provides direction and motivation.",
        "A well-crafted resume opens doors to interviews.",
        "Interview preparation increases confidence and success.",
        "Continuous skill development keeps careers relevant.",
        "Work-life balance contributes to long-term wellbeing.",
        "Mentorship accelerates professional growth and learning.",
        # Science Deep (10)
        "Genetics studies how traits pass from parents to offspring.",
        "Neuroscience explores the structure and function of brains.",
        "Ecology examines interactions between organisms and environment.",
        "Particle physics investigates the fundamental constituents of matter.",
        "Oceanography studies the physical and biological ocean.",
        "Meteorology predicts weather patterns and atmospheric changes.",
        "Paleontology reconstructs life from fossil records.",
        "Thermodynamics governs energy transfer and transformation.",
        "Electromagnetism unifies electric and magnetic phenomena.",
        "Optics studies the behavior and properties of light.",
        # Business (10)
        "Startups drive innovation through disruptive business models.",
        "Supply chains connect producers to consumers globally.",
        "Marketing strategies target specific customer segments.",
        "Financial planning ensures long-term fiscal stability.",
        "Brand identity differentiates products in crowded markets.",
        "Customer feedback drives continuous product improvement.",
        "E-commerce has transformed retail shopping habits.",
        "Risk management protects businesses from uncertainties.",
        "Corporate culture shapes employee behavior and values.",
        "Mergers and acquisitions reshape industry landscapes.",
        # Medicine (10)
        "Vaccines stimulate the immune system to prevent disease.",
        "Antibiotics fight bacterial infections but not viruses.",
        "Diabetes management requires monitoring blood sugar levels.",
        "Cancer screening enables early detection and treatment.",
        "Mental health treatment combines therapy and medication.",
        "Surgery has advanced with minimally invasive techniques.",
        "Public health campaigns promote disease prevention.",
        "Nutrition science guides healthy dietary recommendations.",
        "First aid knowledge can save lives in emergencies.",
        "Medical imaging technologies enable non-invasive diagnosis.",
        # Global Issues (10)
        "International cooperation addresses global challenges together.",
        "Poverty reduction requires economic and social interventions.",
        "Human rights are fundamental freedoms for all people.",
        "Gender equality empowers women and benefits society.",
        "Water scarcity affects billions of people worldwide.",
        "Urbanization creates both opportunities and challenges.",
        "Migration shapes cultural and economic landscapes.",
        "Digital divide limits access to information technology.",
        "Aging populations require adapted healthcare systems.",
        "Disaster preparedness reduces human and economic losses.",
        # Arts Culture (10)
        "Renaissance art celebrated human beauty and potential.",
        "Jazz music originated from African American communities.",
        "Theater performances bring stories to life on stage.",
        "Sculpture transforms raw materials into expressive forms.",
        "Literary fiction explores the depths of human experience.",
        "Folk traditions preserve cultural heritage across generations.",
        "Animation combines artistry with technical innovation.",
        "Design thinking applies creative problem-solving methods.",
        "Cultural festivals celebrate community identity and values.",
        "Independent films offer unique artistic perspectives.",
        # Food Cuisine (8)
        "Italian cuisine celebrates fresh ingredients and simplicity.",
        "Spices transform ordinary dishes into flavorful experiences.",
        "Baking requires precise measurements and temperature control.",
        "Street food offers authentic local culinary experiences.",
        "Fermentation preserves food and creates complex flavors.",
        "Wine tasting involves evaluating aroma body and finish.",
        "Sushi represents the artistry of Japanese cuisine.",
        "Chocolate making is both science and craftsmanship.",
        # Sports Recreation (8)
        "Rock climbing challenges both physical and mental strength.",
        "Cycling is an eco-friendly mode of transportation.",
        "Surfing requires balance timing and wave knowledge.",
        "Chess develops strategic thinking and pattern recognition.",
        "Hiking connects people with nature and fresh air.",
        "Skateboarding culture values creativity and persistence.",
        "Martial arts teach discipline and self-defense skills.",
        "Skiing techniques vary for different snow conditions.",
        # Energy (7)
        "Solar panels convert sunlight into electrical energy.",
        "Wind turbines harness the power of moving air.",
        "Nuclear fission releases enormous amounts of energy.",
        "Hydroelectric dams generate power from flowing water.",
        "Geothermal energy taps into heat beneath the surface.",
        "Battery technology is crucial for renewable energy storage.",
        "Energy efficiency reduces consumption and environmental impact.",
        # Languages (9)
        "Learning Spanish opens doors to Latin American culture.",
        "French is known as the language of diplomacy and love.",
        "Mandarin Chinese has the most native speakers worldwide.",
        "Arabic script is written from right to left beautifully.",
        "German compound words can be remarkably descriptive.",
        "Japanese uses three writing systems: kanji, hiragana, katakana.",
        "Polyglots can speak multiple languages fluently.",
        "Sign language enables communication for the deaf community.",
        "Language acquisition in childhood is remarkably efficient.",
        # Cities (9)
        "Tokyo blends ultramodern architecture with ancient traditions.",
        "London is a global hub for finance and culture.",
        "New York City never sleeps and always inspires.",
        "Singapore is a model of urban planning and efficiency.",
        "Berlin has a vibrant art and startup scene.",
        "Istanbul straddles two continents with rich history.",
        "Bangkok offers incredible street food and temples.",
        "Sydney is famous for its opera house and beaches.",
        "Mumbai pulses with energy as India's commercial capital.",
    ],
    "code": [
        # Functions (12)
        "def hello(): print('world')",
        "def add(a, b): return a + b",
        "def multiply(x, y): return x * y",
        "def is_even(n): return n % 2 == 0",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "def reverse(s): return s[::-1]",
        "def max_of_list(lst): return max(lst)",
        "def count_words(text): return len(text.split())",
        "def fib(n): a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a",
        "def sort_list(items): return sorted(items)",
        "def check_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "def gcd(a, b): return a if b == 0 else gcd(b, a % b)",
        # Classes (12)
        "class Neuron: def forward(self, x): return x",
        "class ResonanceField: def reset(self): pass",
        "class Cortex: def think(self, x): return x",
        "class FeedEngine: def feed(self, data): pass",
        "class SleepEngine: def sleep(self): pass",
        "class Ensemble: def forward(self, x): return x",
        "class PlayEngine: def play(self): pass",
        "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)",
        "class Queue:\n    def __init__(self): self.items = []\n    def enqueue(self, x): self.items.append(x)",
        "class Point:\n    def __init__(self, x, y): self.x = x; self.y = y",
        "class Vector:\n    def __init__(self, data): self.data = data\n    def sum(self): return sum(self.data)",
        "class Logger:\n    def info(self, msg): print(f'[INFO] {msg}')",
        # Utility (10)
        "import torch; print(torch.__version__)",
        "import json; data = json.loads('{\"key\": \"value\"}')",
        "import os; files = os.listdir('.')",
        "import sys; print(sys.path)",
        "import math; result = math.sqrt(16)",
        "import random; choice = random.choice([1,2,3])",
        "import datetime; now = datetime.datetime.now()",
        "import collections; counter = collections.Counter('abracadabra')",
        "import itertools; pairs = list(itertools.combinations([1,2,3], 2))",
        "import functools; result = functools.reduce(lambda a,b: a+b, [1,2,3,4])",
        # Patterns (10)
        "def generate(prompt): return prompt + ' generated'",
        "def validate_input(x): assert x is not None",
        "def save_state(model, path): torch.save(model, path)",
        "def load_state(path): return torch.load(path)",
        "def encode(text, tokenizer): return tokenizer.encode(text)",
        "def decode(ids, tokenizer): return tokenizer.decode(ids)",
        "def train(model, data): model.fit(data)",
        "def predict(model, x): return model(x)",
        "def loss_fn(pred, target): return (pred - target).mean()",
        "def accuracy(preds, labels): return (preds == labels).float().mean()",
        # Algorithms (8)
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]",
        "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))",
        "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quick_sort([x for x in arr[1:] if x < pivot]) + [pivot] + quick_sort([x for x in arr[1:] if x >= pivot])",
        "def linear_search(arr, target):\n    for i, x in enumerate(arr):\n        if x == target: return i\n    return -1",
        "def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i-1\n        while j >= 0 and arr[j] > key:\n            arr[j+1] = arr[j]\n            j -= 1\n        arr[j+1] = key",
        "def depth_first_search(graph, start, visited=None):\n    if visited is None: visited = set()\n    visited.add(start)\n    for next in graph[start] - visited:\n        depth_first_search(graph, next, visited)",
        "def breadth_first_search(graph, start):\n    visited = set([start])\n    queue = [start]\n    while queue:\n        vertex = queue.pop(0)\n        for next in graph[vertex] - visited:\n            visited.add(next)\n            queue.append(next)",
        # Data Structures (10)
        "class ListNode:\n    def __init__(self, val=0, next=None): self.val = val; self.next = next",
        "class TreeNode:\n    def __init__(self, val=0, left=None, right=None): self.val = val; self.left = left; self.right = right",
        "class HashTable:\n    def __init__(self, size=100): self.size = size; self.table = [[] for _ in range(size)]",
        "class Graph:\n    def __init__(self): self.adj = {}\n    def add_edge(self, u, v): self.adj.setdefault(u, []).append(v)",
        "class Deque:\n    def __init__(self): self.items = []\n    def push_front(self, x): self.items.insert(0, x)",
        "class MinHeap:\n    def __init__(self): self.heap = []\n    def push(self, x): self.heap.append(x); self._sift_up(len(self.heap)-1)",
        "class TrieNode:\n    def __init__(self): self.children = {}; self.is_end = False",
        "class CircularQueue:\n    def __init__(self, k): self.k = k; self.q = [0]*k; self.front = self.rear = -1",
        "class LinkedList:\n    def __init__(self): self.head = None\n    def append(self, val): pass",
        "class BinarySearchTree:\n    def __init__(self): self.root = None\n    def insert(self, val): pass",
        # Sorting (8)
        "def selection_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        min_idx = i\n        for j in range(i+1, n):\n            if arr[j] < arr[min_idx]: min_idx = j\n        arr[i], arr[min_idx] = arr[min_idx], arr[i]",
        "def heap_sort(arr):\n    import heapq\n    heapq.heapify(arr)\n    return [heapq.heappop(arr) for _ in range(len(arr))]",
        "def counting_sort(arr, max_val):\n    count = [0] * (max_val + 1)\n    for x in arr: count[x] += 1\n    result = []\n    for i, c in enumerate(count): result.extend([i]*c)\n    return result",
        "def radix_sort(arr):\n    max_digit = len(str(max(arr)))\n    for d in range(max_digit):\n        buckets = [[] for _ in range(10)]\n        for x in arr: buckets[(x//10**d)%10].append(x)\n        arr = [y for b in buckets for y in b]\n    return arr",
        "def shell_sort(arr):\n    n = len(arr); gap = n // 2\n    while gap > 0:\n        for i in range(gap, n):\n            temp = arr[i]; j = i\n            while j >= gap and arr[j-gap] > temp: arr[j] = arr[j-gap]; j -= gap\n            arr[j] = temp\n        gap //= 2",
        "def bucket_sort(arr):\n    buckets = [[] for _ in range(len(arr))]\n    for x in arr: buckets[int(x*len(arr))].append(x)\n    return [y for b in buckets for y in sorted(b)]",
        "def tim_sort(arr): return sorted(arr)  # Python's built-in Timsort",
        "def pancake_sort(arr):\n    def flip(arr, k): arr[:k] = arr[:k][::-1]\n    n = len(arr)\n    for size in range(n, 1, -1):\n        max_idx = arr.index(max(arr[:size]))\n        flip(arr, max_idx+1); flip(arr, size)",
        # Error Handling (8)
        "def safe_divide(a, b):\n    try: return a / b\n    except ZeroDivisionError: return float('inf')\n    except TypeError: return None",
        "def parse_json(data):\n    try: return json.loads(data)\n    except json.JSONDecodeError as e: print(f'Invalid JSON: {e}'); return None",
        "def read_file(path):\n    try:\n        with open(path) as f: return f.read()\n    except FileNotFoundError: return ''\n    except PermissionError: return None",
        "def validate_range(x, lo, hi):\n    if not isinstance(x, (int, float)): raise TypeError('Must be number')\n    if x < lo or x > hi: raise ValueError(f'{x} not in [{lo},{hi}]')",
        "class Retry:\n    def __init__(self, max_retries=3): self.max = max_retries\n    def call(self, fn, *args):\n        for i in range(self.max):\n            try: return fn(*args)\n            except Exception: pass\n        raise RuntimeError('Max retries exceeded')",
        "def atomic_write(path, content):\n    tmp = path + '.tmp'\n    write(tmp, content)\n    os.replace(tmp, path)  # atomic on POSIX",
        "def with_timeout(fn, seconds, default=None):\n    import signal\n    def handler(signum, frame): raise TimeoutError()\n    signal.signal(signal.SIGALRM, handler)\n    signal.alarm(seconds)\n    try: return fn()\n    except TimeoutError: return default\n    finally: signal.alarm(0)",
        "class Result:\n    def __init__(self, value=None, error=None): self.value = value; self.error = error\n    def is_ok(self): return self.error is None\n    def unwrap(self): return self.value if self.is_ok() else raise Exception(self.error)",
        # Concurrency (8)
        "import threading\nlock = threading.Lock()\nwith lock:\n    print('critical section')",
        "from concurrent.futures import ThreadPoolExecutor\nexecutor = ThreadPoolExecutor(max_workers=4)\nfutures = [executor.submit(task, arg) for arg in args]",
        "import asyncio\nasync def async_fetch(url):\n    await asyncio.sleep(1)\n    return f'result from {url}'",
        "from multiprocessing import Pool\nwith Pool(processes=4) as pool:\n    results = pool.map(process_item, items)",
        "queue = Queue()\nthread_a = Thread(target=producer, args=(queue,))\nthread_b = Thread(target=consumer, args=(queue,))",
        "semaphore = threading.Semaphore(5)\nwith semaphore:\n    limited_concurrent_operation()",
        "import queue\npq = queue.PriorityQueue()\npq.put((1, 'low')); pq.put((0, 'high'))\nwhile not pq.empty(): print(pq.get())",
        "async def gather_tasks():\n    results = await asyncio.gather(\n        fetch_page('url1'),\n        fetch_page('url2'),\n        fetch_page('url3'),\n    )\n    return results",
        # Testing (8)
        "import unittest\nclass TestMath(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(1, 2), 3)\n    def test_divide_by_zero(self):\n        with self.assertRaises(ZeroDivisionError): divide(1, 0)",
        "import pytest\ndef test_reverse():\n    assert reverse('abc') == 'cba'\n    assert reverse('') == ''\n\n@pytest.mark.parametrize('n,expected', [(1,1), (2,1), (5,5)])",
        "def test_edge_cases():\n    # Test empty input\n    result = process([])\n    assert result == []\n    # Test large input\n    result = process(list(range(100000)))\n    assert len(result) == 100000",
        "from unittest.mock import Mock, patch\ndef test_with_mock():\n    api = Mock()\n    api.fetch.return_value = {'data': [1,2,3]}\n    assert len(api.fetch()['data']) == 3",
        "def test_concurrently():\n    results = []\n    threads = [Thread(target=lambda: results.append(fn())) for _ in range(10)]\n    for t in threads: t.start()\n    for t in threads: t.join()\n    assert all(r is not None for r in results)",
        "def benchmark(fn, iterations=1000):\n    import time\n    start = time.perf_counter()\n    for _ in range(iterations): fn()\n    elapsed = time.perf_counter() - start\n    return elapsed / iterations",
        "class TestDatabase(unittest.TestCase):\n    def setUp(self): self.db = connect(':memory:')\n    def tearDown(self): self.db.close()\n    def test_insert(self):\n        self.db.execute('INSERT INTO t VALUES (1)')\n        rows = self.db.execute('SELECT * FROM t').fetchall()\n        self.assertEqual(len(rows), 1)",
        "def test_property_based():\n    from hypothesis import given, strategies as st\n    @given(st.integers(), st.integers())\n    def test_commutative(a, b):\n        assert add(a, b) == add(b, a)",
        # Design Patterns (8)
        "class Singleton:\n    _instance = None\n    def __new__(cls, *args, **kwargs):\n        if cls._instance is None: cls._instance = super().__new__(cls)\n        return cls._instance",
        "class Observer:\n    def __init__(self): self._observers = []\n    def subscribe(self, observer): self._observers.append(observer)\n    def notify(self, event):\n        for o in self._observers: o.on_event(event)",
        "class Context: pass\nclass StrategyA: pass\nclass StrategyB: pass\nctx = Context()\nctx.strategy = StrategyA() if condition else StrategyB()",
        "class Handler:\n    def __init__(self, next_handler=None): self.next = next_handler\n    def handle(self, request):\n        if self.can_handle(request): return self.process(request)\n        return self.next.handle(request) if self.next else None",
        "class Command:\n    def execute(self): raise NotImplementedError\nclass PrintCommand(Command):\n    def __init__(self, text): self.text = text\n    def execute(self): print(self.text)",
        "class Component:\n    def operation(self): pass\nclass Decorator(Component):\n    def __init__(self, component): self._component = component\n    def operation(self): return f'[{self._component.operation()}]'",
        "class Iterator:\n    def __init__(self, collection): self._collection = collection; self._index = 0\n    def __next__(self):\n        if self._index >= len(self._collection): raise StopIteration\n        item = self._collection[self._index]; self._index += 1\n        return item",
        "class State:\n    def handle(self, context): pass\nclass StateA(State): pass\nclass StateB(State): pass\nclass Machine:\n    def __init__(self): self.state = StateA()\n    def transition(self, new_state): self.state = new_state",
        # File IO (8)
        "def write_lines(path, lines):\n    with open(path, 'w', encoding='utf-8') as f:\n        f.write('\\n'.join(lines))",
        "def read_csv(path):\n    import csv\n    with open(path, newline='') as f:\n        return list(csv.reader(f))",
        "def walk_directory(root):\n    for dirpath, dirnames, filenames in os.walk(root):\n        for fn in filenames:\n            yield os.path.join(dirpath, fn)",
        "def read_chunks(path, chunk_size=8192):\n    with open(path, 'rb') as f:\n        while chunk := f.read(chunk_size): yield chunk",
        "def append_log(path, message):\n    with open(path, 'a') as f:\n        f.write(f'{datetime.now().isoformat()} {message}\\n')",
        "import tempfile\nwith tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:\n    f.write('temporary data')\n    tmp_path = f.name",
        "def read_config(path):\n    import configparser\n    cfg = configparser.ConfigParser()\n    cfg.read(path)\n    return {s: dict(cfg.items(s)) for s in cfg.sections()}",
        "def count_lines(path):\n    with open(path) as f:\n        return sum(1 for _ in f)",
        # Regex (7)
        "import re\npattern = r'\\b\\w+@\\w+\\.\\w+\\b'\nemails = re.findall(pattern, text)",
        "phone_pattern = r'\\b1[3-9]\\d{9}\\b'\nphones = re.findall(phone_pattern, document)",
        "result = re.sub(r'\\s+', ' ', text)  # collapse whitespace",
        "match = re.search(r'version=(\\d+\\.\\d+)', config_str)\nversion = match.group(1) if match else 'unknown'",
        "tokens = re.split(r'[,;]+', csv_line)  # split on commas or semicolons",
        "url_pattern = r'https?://[\\w.-]+(?:\\.[\\w.-]+)+[\\w\\-._~:/?#@!$&()*+,;=]+'\nurls = re.findall(url_pattern, html_content)",
        "def is_valid_ip(s):\n    pattern = r'^(\\d{1,3}\\.){3}\\d{1,3}$'\n    return bool(re.match(pattern, s))",
        # Networking (8)
        "import requests\nresponse = requests.get('https://api.example.com/data')\ndata = response.json()",
        "import socket\nsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nsock.connect(('localhost', 8080))\nsock.send(b'Hello')\nsock.close()",
        "from http.server import HTTPServer, BaseHTTPRequestHandler\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.send_header('Content-Type', 'application/json')\n        self.end_headers()\n        self.wfile.write(b'{\"status\":\"ok\"}')",
        "import urllib.parse\nparams = {'q': 'search term', 'page': '1'}\nencoded = urllib.parse.urlencode(params)\nurl = f'https://example.com/search?{encoded}'",
        "import json\nresponse_data = {'users': [{'id': 1, 'name': 'Alice'}]}\njson_str = json.dumps(response_data, indent=2)",
        "headers = {'Authorization': 'Bearer token123', 'Content-Type': 'application/json'}\nresponse = requests.post(url, headers=headers, json={'key': 'value'})",
        "def download_file(url, dest):\n    r = requests.get(url, stream=True)\n    with open(dest, 'wb') as f:\n        for chunk in r.iter_content(chunk_size=8192):\n            f.write(chunk)",
        "import websockets\nasync def ws_client():\n    async with websockets.connect('ws://localhost:8765') as ws:\n        await ws.send('Hello')\n        reply = await ws.recv()\n        return reply",
        # Database (10)
        "import sqlite3\nconn = sqlite3.connect(':memory:')\ncursor = conn.cursor()\ncursor.execute('CREATE TABLE users (id INTEGER, name TEXT)')",
        "def query_db(query, params=()):\n    cursor.execute(query, params)\n    return cursor.fetchall()",
        "class Database:\n    def __init__(self, path): self.conn = sqlite3.connect(path)\n    def execute(self, sql, params=()): return self.conn.execute(sql, params)",
        "class ORM:\n    def __init__(self, model): self.model = model\n    def filter(self, **kwargs):\n        conditions = ' AND '.join(f'{k}=?' for k in kwargs)\n        return f'SELECT * FROM {self.model} WHERE {conditions}'",
        "def migrate(db, schema):\n    for table, columns in schema.items():\n        cols = ', '.join(f'{c} {t}' for c, t in columns.items())\n        db.execute(f'CREATE TABLE IF NOT EXISTS {table} ({cols})')",
        "def transaction(db, operations):\n    try:\n        for op in operations: db.execute(op)\n        db.commit()\n    except Exception:\n        db.rollback()\n        raise",
        "class ConnectionPool:\n    def __init__(self, max_conn=10): self.pool = [create_conn() for _ in range(max_conn)]\n    def acquire(self): return self.pool.pop()\n    def release(self, conn): self.pool.append(conn)",
        "def create_index(db, table, column):\n    db.execute(f'CREATE INDEX idx_{table}_{column} ON {table}({column})')",
        "class Migration:\n    def up(self): raise NotImplementedError\n    def down(self): raise NotImplementedError\nclass AddEmailColumn(Migration):\n    def up(self): db.execute('ALTER TABLE users ADD COLUMN email TEXT')",
        "def batch_insert(db, table, rows):\n    placeholders = ', '.join(['?'] * len(rows[0]))\n    db.executemany(f'INSERT INTO {table} VALUES ({placeholders})', rows)",
        # API Design (10)
        "from flask import Flask, request, jsonify\napp = Flask(__name__)\n@app.route('/api/users', methods=['GET'])\ndef get_users():\n    return jsonify({'users': [{'id': 1, 'name': 'Alice'}]})",
        "class RESTAPI:\n    def get(self, resource, id=None): pass\n    def post(self, resource, data): pass\n    def put(self, resource, id, data): pass\n    def delete(self, resource, id): pass",
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/items/{item_id}')\nasync def read_item(item_id: int):\n    return {'item_id': item_id}",
        "class APIRateLimiter:\n    def __init__(self, max_req=100, window=60): self.requests = {}\n    def allow(self, client_id):\n        now = time.time()\n        self.requests.setdefault(client_id, []).append(now)\n        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < self.window]\n        return len(self.requests[client_id]) <= self.max_req",
        "def paginate(query, page=1, per_page=20):\n    offset = (page - 1) * per_page\n    return query.limit(per_page).offset(offset).all()",
        "class APIKeyAuth:\n    def __init__(self): self.keys = {}\n    def generate_key(self, user_id):\n        key = secrets.token_hex(32)\n        self.keys[key] = user_id\n        return key\n    def validate(self, key): return self.keys.get(key)",
        "def error_response(code, message):\n    return jsonify({'error': {'code': code, 'message': message}}), code",
        "class APIVersioning:\n    def __init__(self): self.versions = {'v1': v1_routes, 'v2': v2_routes}\n    def route(self, version, path): return self.versions[version][path]",
        "from graphql import build_schema, graphql_sync\nschema = build_schema('''type Query { hello: String }''')\nresult = graphql_sync(schema, '{ hello }')",
        "class Webhook:\n    def __init__(self, url, secret): self.url = url; self.secret = secret\n    def send(self, event, data):\n        payload = json.dumps({'event': event, 'data': data})\n        signature = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()\n        return requests.post(self.url, data=payload, headers={'X-Signature': signature})",
        # Optimization (10)
        "from functools import lru_cache\n@lru_cache(maxsize=128)\ndef expensive_computation(n):\n    return sum(i*i for i in range(n))",
        "def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache: cache[args] = fn(*args)\n        return cache[args]\n    return wrapper",
        "def profile(fn):\n    import cProfile, pstats\n    profiler = cProfile.Profile()\n    profiler.enable()\n    result = fn()\n    profiler.disable()\n    pstats.Stats(profiler).sort_stats('cumtime').print_stats(10)\n    return result",
        "class ObjectPool:\n    def __init__(self, factory, max_size=20):\n        self.factory = factory\n        self.pool = [factory() for _ in range(max_size)]\n    def acquire(self): return self.pool.pop() if self.pool else self.factory()\n    def release(self, obj): self.pool.append(obj)",
        "def vectorize_operation(data, operation):\n    import numpy as np\n    arr = np.array(data)\n    return operation(arr)",
        "def parallel_map(fn, items, workers=4):\n    from concurrent.futures import ProcessPoolExecutor\n    with ProcessPoolExecutor(max_workers=workers) as executor:\n        return list(executor.map(fn, items))",
        "def cached_property(fn):\n    name = fn.__name__\n    def getter(self):\n        if name not in self.__dict__: self.__dict__[name] = fn(self)\n        return self.__dict__[name]\n    return property(getter)",
        "class LazyLoader:\n    def __init__(self, loader): self._loader = loader; self._value = None\n    def __call__(self):\n        if self._value is None: self._value = self._loader()\n        return self._value",
        "def batch_process(items, batch_size=64, fn=None):\n    for i in range(0, len(items), batch_size):\n        batch = items[i:i+batch_size]\n        yield [fn(item) for item in batch]",
        "def sliding_window(iterable, size=3):\n    from collections import deque\n    window = deque(maxlen=size)\n    for item in iterable:\n        window.append(item)\n        if len(window) == size: yield tuple(window)",
        # Security (10)
        "import hashlib\ndef hash_password(password, salt=None):\n    if salt is None: salt = os.urandom(16)\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000), salt",
        "import secrets\ndef generate_token(length=32):\n    return secrets.token_hex(length)",
        "import re\ndef sanitize_html(text):\n    return re.sub(r'<[^>]*>', '', text)  # strip HTML tags to prevent XSS",
        "class CSRFProtection:\n    def __init__(self): self.tokens = {}\n    def generate(self, session_id):\n        token = secrets.token_hex(16)\n        self.tokens[session_id] = token\n        return token\n    def validate(self, session_id, token): return self.tokens.get(session_id) == token",
        "def validate_jwt(token, secret):\n    import jwt\n    try: return jwt.decode(token, secret, algorithms=['HS256'])\n    except jwt.InvalidTokenError: return None",
        "class RateLimiter:\n    def __init__(self, max_requests, window_seconds):\n        self.max = max_requests; self.window = window_seconds; self.records = {}\n    def is_allowed(self, key):\n        now = time.time()\n        self.records.setdefault(key, [])\n        self.records[key] = [t for t in self.records[key] if now - t < self.window]\n        if len(self.records[key]) >= self.max: return False\n        self.records[key].append(now); return True",
        "def encrypt_aes(data, key):\n    from Crypto.Cipher import AES\n    cipher = AES.new(key, AES.MODE_GCM)\n    ciphertext, tag = cipher.encrypt_and_digest(data.encode())\n    return cipher.nonce + tag + ciphertext",
        "def check_sql_injection(query):\n    dangerous = ['DROP', 'DELETE', '--', 'UNION', '1=1']\n    upper = query.upper()\n    return any(pattern in upper for pattern in dangerous)",
        "class SecureSession:\n    def __init__(self, secret_key): self.key = secret_key\n    def create(self, user_id):\n        payload = {'user_id': user_id, 'exp': time.time() + 3600}\n        return jwt.encode(payload, self.key, algorithm='HS256')\n    def verify(self, token):\n        try: return jwt.decode(token, self.key, algorithms=['HS256'])\n        except: return None",
        "def generate_self_signed_cert():\n    from cryptography import x509\n    from cryptography.x509.oid import NameOID\n    from cryptography.hazmat.primitives import hashes, serialization\n    from cryptography.hazmat.primitives.asymmetric import rsa\n    key = rsa.generate_private_key(65537, 2048)\n    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])\n    cert = x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key()).serial_number(x509.random_serial_number()).sign(key, hashes.SHA256())\n    return cert.public_bytes(serialization.Encoding.PEM)",
        # Data Science (10)
        "import numpy as np\ndata = np.array([1, 2, 3, 4, 5])\nmean = np.mean(data)\nstd = np.std(data)",
        "import pandas as pd\ndf = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})\ndf['C'] = df['A'] + df['B']",
        "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)",
        "import matplotlib.pyplot as plt\nplt.plot([1,2,3,4], [1,4,9,16])\nplt.xlabel('x')\nplt.ylabel('y')\nplt.title('Quadratic Function')",
        "def normalize(data):\n    arr = np.array(data)\n    return (arr - arr.mean()) / arr.std()",
        "from scipy import stats\nt_stat, p_value = stats.ttest_ind(group_a, group_b)",
        "def one_hot_encode(labels):\n    unique = sorted(set(labels))\n    return [[1 if label == u else 0 for u in unique] for label in labels]",
        "from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)",
        "def confusion_matrix(y_true, y_pred, labels):\n    n = len(labels)\n    cm = [[0]*n for _ in range(n)]\n    for t, p in zip(y_true, y_pred): cm[t][p] += 1\n    return cm",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score\nacc = accuracy_score(y_true, y_pred)\nprec = precision_score(y_true, y_pred)\nrec = recall_score(y_true, y_pred)",
        # CLI Tools (8)
        "import argparse\nparser = argparse.ArgumentParser(description='Process files')\nparser.add_argument('input', help='input file path')\nparser.add_argument('-o', '--output', default='output.txt', help='output file')\nparser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')\nargs = parser.parse_args()",
        "import sys\ndef main():\n    if len(sys.argv) < 2:\n        print('Usage: script.py <filename>', file=sys.stderr)\n        sys.exit(1)\n    process_file(sys.argv[1])",
        "import click\n@click.command()\n@click.argument('filename')\n@click.option('--count', default=1, help='number of times')\ndef repeat(filename, count):\n    with open(filename) as f:\n        content = f.read()\n    for _ in range(count): print(content)",
        "import typer\napp = typer.Typer()\n@app.command()\ndef hello(name: str = 'World'):\n    typer.echo(f'Hello {name}')",
        "class ProgressBar:\n    def __init__(self, total, width=40): self.total = total; self.width = width; self.current = 0\n    def update(self, n=1):\n        self.current += n\n        pct = self.current / self.total\n        filled = int(self.width * pct)\n        bar = '=' * filled + '-' * (self.width - filled)\n        print(f'\\r[{bar}] {pct:.0%}', end='', flush=True)",
        "def confirm_action(prompt):\n    response = input(f'{prompt} [y/N] ').strip().lower()\n    return response in ('y', 'yes')",
        "import colorama\nfrom colorama import Fore, Style\ncolorama.init()\nprint(f'{Fore.GREEN}Success!{Style.RESET_ALL}')\nprint(f'{Fore.RED}Error!{Style.RESET_ALL}')",
        "class TableFormatter:\n    def __init__(self, headers): self.headers = headers; self.rows = []\n    def add_row(self, row): self.rows.append(row)\n    def render(self):\n        col_widths = [max(len(str(r[i])) for r in [self.headers] + self.rows) for i in range(len(self.headers))]\n        sep = '+' + '+'.join('-'*(w+2) for w in col_widths) + '+'\n        lines = [sep]\n        lines.append('|' + '|'.join(f' {h:<{w}} ' for h, w in zip(self.headers, col_widths)) + '|')\n        lines.append(sep)\n        for row in self.rows:\n            lines.append('|' + '|'.join(f' {str(v):<{w}} ' for v, w in zip(row, col_widths)) + '|')\n        lines.append(sep)\n        return '\\n'.join(lines)",
        # Machine Learning (7)
        "import torch\nimport torch.nn as nn\nclass SimpleNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(784, 128)\n        self.fc2 = nn.Linear(128, 10)\n    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))",
        "class SGD:\n    def __init__(self, params, lr=0.01): self.params = list(params); self.lr = lr\n    def step(self):\n        for p in self.params:\n            if p.grad is not None: p.data -= self.lr * p.grad",
        "def cross_entropy_loss(logits, targets):\n    exp_logits = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)\n    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)\n    return -torch.log(probs.gather(-1, targets.unsqueeze(-1))).mean()",
        "class DataLoader:\n    def __init__(self, data, batch_size=32, shuffle=True):\n        self.data = data; self.batch_size = batch_size; self.shuffle = shuffle\n    def __iter__(self):\n        indices = list(range(len(self.data)))\n        if self.shuffle: random.shuffle(indices)\n        for i in range(0, len(indices), self.batch_size):\n            yield [self.data[j] for j in indices[i:i+self.batch_size]]",
        "def kfold_split(data, k=5):\n    fold_size = len(data) // k\n    for i in range(k):\n        test = data[i*fold_size:(i+1)*fold_size]\n        train = data[:i*fold_size] + data[(i+1)*fold_size:]\n        yield train, test",
        "def gradient_clip(params, max_norm):\n    total_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))\n    if total_norm > max_norm:\n        scale = max_norm / (total_norm + 1e-6)\n        for p in params:\n            if p.grad is not None: p.grad *= scale",
        "class EarlyStopping:\n    def __init__(self, patience=5): self.patience = patience; self.best = float('inf'); self.counter = 0\n    def __call__(self, val_loss):\n        if val_loss < self.best: self.best = val_loss; self.counter = 0; return False\n        self.counter += 1; return self.counter >= self.patience",
        # DevOps (7)
        "def deploy(service, version, env='production'):\n    config = load_config(env)\n    image = build_image(service, version)\n    push_image(image, config.registry)\n    update_service(service, image, config)\n    return wait_for_healthy(service, timeout=300)",
        "class Dockerfile:\n    @staticmethod\n    def generate(base_image, commands, entrypoint):\n        lines = [f'FROM {base_image}']\n        for cmd in commands: lines.append(f'RUN {cmd}')\n        lines.append(f'ENTRYPOINT {json.dumps(entrypoint)}')\n        return '\\n'.join(lines)",
        "def health_check(endpoint, max_retries=10, interval=3):\n    for i in range(max_retries):\n        try:\n            r = requests.get(endpoint)\n            if r.status_code == 200: return True\n        except: pass\n        time.sleep(interval)\n    return False",
        "class ConfigManager:\n    def __init__(self): self.configs = {}\n    def load(self, env):\n        with open(f'config/{env}.yaml') as f: self.configs[env] = yaml.safe_load(f)\n    def get(self, key, env='production', default=None): return self.configs.get(env, {}).get(key, default)",
        "def monitor_metrics():\n    import psutil\n    return {\n        'cpu_percent': psutil.cpu_percent(),\n        'memory_percent': psutil.virtual_memory().percent,\n        'disk_percent': psutil.disk_usage('/').percent,\n    }",
        "class ServiceDiscovery:\n    def __init__(self): self.services = {}\n    def register(self, name, host, port):\n        self.services[name] = {'host': host, 'port': port, 'last_heartbeat': time.time()}\n    def discover(self, name): return self.services.get(name)",
        "def rollback(service, previous_version):\n    logger.warning(f'Rolling back {service} to {previous_version}')\n    update_service(service, f'{service}:{previous_version}')\n    if not health_check(f'http://{service}/health'):\n        raise RuntimeError(f'Rollback of {service} failed')",
        # Functional Programming (8)
        "from functools import reduce\ndef pipe(*functions):\n    return reduce(lambda f, g: lambda x: g(f(x)), functions)",
        "def curry(fn):\n    from functools import partial\n    def curried(*args, **kwargs):\n        if len(args) + len(kwargs) >= fn.__code__.co_argcount:\n            return fn(*args, **kwargs)\n        return partial(curried, *args, **kwargs)\n    return curried",
        "class Maybe:\n    def __init__(self, value): self.value = value\n    def map(self, fn): return Maybe(None) if self.value is None else Maybe(fn(self.value))\n    def get_or(self, default): return self.value if self.value is not None else default\n    @staticmethod\n    def of(value): return Maybe(value)",
        "class Either:\n    def __init__(self, left=None, right=None): self.left = left; self.right = right\n    def is_left(self): return self.left is not None\n    def is_right(self): return self.right is not None\n    def map(self, fn): return Either(right=fn(self.right)) if self.is_right() else self\n    @staticmethod\n    def left(value): return Either(left=value)\n    @staticmethod\n    def right(value): return Either(right=value)",
        "def compose(*functions):\n    def composed(x):\n        for f in reversed(functions): x = f(x)\n        return x\n    return composed",
        "class LazySeq:\n    def __init__(self, generator): self._gen = generator\n    def take(self, n):\n        result = []\n        for i, x in enumerate(self._gen()):\n            if i >= n: break\n            result.append(x)\n        return result\n    def map(self, fn): return LazySeq(lambda: (fn(x) for x in self._gen()))\n    def filter(self, pred): return LazySeq(lambda: (x for x in self._gen() if pred(x)))",
        "def memoized(fn):\n    cache = {}\n    def wrapper(*args, **kwargs):\n        key = (args, tuple(sorted(kwargs.items())))\n        if key not in cache: cache[key] = fn(*args, **kwargs)\n        return cache[key]\n    wrapper.cache = cache\n    return wrapper",
        "class ImmutableDict:\n    def __init__(self, data=None): self._data = dict(data or {})\n    def __getitem__(self, key): return self._data[key]\n    def __setitem__(self, key, value): raise TypeError('ImmutableDict does not support item assignment')\n    def set(self, key, value): return ImmutableDict({**self._data, key: value})\n    def items(self): return self._data.items()",
        # Web Scraping (8)
        "from bs4 import BeautifulSoup\nsoup = BeautifulSoup(html, 'html.parser')\ntitles = [h.get_text() for h in soup.find_all('h2')]",
        "import scrapy\nclass MySpider(scrapy.Spider):\n    name = 'myspider'\n    start_urls = ['https://example.com']\n    def parse(self, response):\n        for item in response.css('div.item'):\n            yield {'title': item.css('h3::text').get()}",
        "import selenium\nfrom selenium import webdriver\ndriver = webdriver.Chrome()\ndriver.get('https://example.com')\nelement = driver.find_element(By.ID, 'main')\ndriver.quit()",
        "def fetch_with_retry(url, max_retries=3, backoff=2):\n    for attempt in range(max_retries):\n        try:\n            r = requests.get(url, timeout=10)\n            if r.status_code == 200: return r\n            if r.status_code == 429: time.sleep(backoff ** attempt)\n        except requests.RequestException:\n            time.sleep(backoff ** attempt)\n    return None",
        "def extract_links(html):\n    pattern = r'href=[\"\\'](.*?)[\"\\']'\n    return re.findall(pattern, html)",
        "def parse_sitemap(url):\n    import xml.etree.ElementTree as ET\n    r = requests.get(url)\n    root = ET.fromstring(r.content)\n    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}\n    return [loc.text for loc in root.findall('.//sm:loc', ns)]",
        "from playwright.sync_api import sync_playwright\nwith sync_playwright() as p:\n    browser = p.chromium.launch()\n    page = browser.new_page()\n    page.goto('https://example.com')\n    content = page.content()\n    browser.close()",
        "def scrape_table(url, table_index=0):\n    import pandas as pd\n    tables = pd.read_html(url)\n    return tables[table_index] if tables else None",
        # Logging Monitoring (8)
        "import logging\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')\nlogger = logging.getLogger(__name__)\nlogger.info('Application started')",
        "class MetricsCollector:\n    def __init__(self): self.metrics = {}\n    def record(self, name, value, tags=None):\n        self.metrics.setdefault(name, []).append({'value': value, 'time': time.time(), 'tags': tags or {}})\n    def summary(self, name):\n        values = [m['value'] for m in self.metrics.get(name, [])]\n        return {'count': len(values), 'avg': sum(values)/len(values) if values else 0}",
        "import structlog\nlog = structlog.get_logger()\nlog.info('user_logged_in', user_id=42, ip='192.168.1.1')",
        "def trace(fn):\n    @functools.wraps(fn)\n    def wrapper(*args, **kwargs):\n        logger.debug(f'Entering {fn.__name__}')\n        result = fn(*args, **kwargs)\n        logger.debug(f'Exiting {fn.__name__}')\n        return result\n    return wrapper",
        "class AlertManager:\n    def __init__(self): self.alerts = []\n    def fire(self, severity, message):\n        alert = {'severity': severity, 'message': message, 'time': time.time()}\n        self.alerts.append(alert)\n        if severity == 'critical': self._notify(alert)",
        "def rotate_logs(log_path, max_size_mb=10, backup_count=5):\n    from logging.handlers import RotatingFileHandler\n    handler = RotatingFileHandler(log_path, maxBytes=max_size_mb*1024*1024, backupCount=backup_count)\n    return handler",
        "def health_check_endpoint(app):\n    @app.route('/health')\n    def health():\n        checks = {'database': check_db(), 'cache': check_redis()}\n        status = all(checks.values())\n        return jsonify({'status': 'ok' if status else 'degraded', 'checks': checks}), 200 if status else 503",
        "class DistributedTracer:\n    def __init__(self, service_name): self.service = service_name\n    def start_span(self, operation):\n        span_id = str(uuid.uuid4())\n        logger.info(f'[trace] {self.service}/{operation} span={span_id}')\n        return span_id\n    def end_span(self, span_id): logger.debug(f'[trace] span={span_id} ended')",
        # Caching (7)
        "from functools import lru_cache\n@lru_cache(maxsize=256)\ndef fetch_user(user_id):\n    return db.query('SELECT * FROM users WHERE id=?', (user_id,))",
        "class TTLCache:\n    def __init__(self, ttl=60): self.cache = {}; self.ttl = ttl\n    def get(self, key):\n        if key in self.cache:\n            value, expiry = self.cache[key]\n            if time.time() < expiry: return value\n            del self.cache[key]\n        return None\n    def set(self, key, value): self.cache[key] = (value, time.time() + self.ttl)",
        "import redis\nr = redis.Redis(host='localhost', port=6379, decode_responses=True)\nr.set('key', 'value', ex=3600)\ncached = r.get('key')",
        "class TwoLevelCache:\n    def __init__(self): self.l1 = {}; self.l2 = redis.Redis()\n    def get(self, key):\n        if key in self.l1: return self.l1[key]\n        val = self.l2.get(key)\n        if val: self.l1[key] = val\n        return val\n    def set(self, key, value, ttl=300):\n        self.l1[key] = value; self.l2.setex(key, ttl, value)",
        "def cache_key(*args, **kwargs):\n    import hashlib, json\n    raw = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)\n    return hashlib.md5(raw.encode()).hexdigest()",
        "class CacheWarmer:\n    def __init__(self, cache, data_loader): self.cache = cache; self.loader = data_loader\n    def warm(self, keys):\n        for key in keys:\n            if self.cache.get(key) is None:\n                self.cache.set(key, self.loader(key))",
        "def cache_decorator(ttl=300):\n    cache = {}\n    def decorator(fn):\n        @functools.wraps(fn)\n        def wrapper(*args, **kwargs):\n            key = (args, tuple(sorted(kwargs.items())))\n            if key in cache:\n                value, expiry = cache[key]\n                if time.time() < expiry: return value\n            result = fn(*args, **kwargs)\n            cache[key] = (result, time.time() + ttl)\n            return result\n        return wrapper\n    return decorator",
        # Data Visualization (8)
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.bar(['A', 'B', 'C'], [3, 7, 2])\nax.set_title('Bar Chart')\nfig.savefig('chart.png')",
        "import seaborn as sns\nsns.set_theme()\nsns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')",
        "def plot_learning_curve(losses, val_losses):\n    plt.figure(figsize=(10, 6))\n    plt.plot(losses, label='Training Loss')\n    plt.plot(val_losses, label='Validation Loss')\n    plt.xlabel('Epoch')\n    plt.ylabel('Loss')\n    plt.legend()\n    plt.grid(True)",
        "import plotly.express as px\nfig = px.scatter(df, x='age', y='income', color='region', size='population')\nfig.write_html('scatter.html')",
        "def create_dashboard(data):\n    from dash import Dash, dcc, html\n    app = Dash(__name__)\n    app.layout = html.Div([\n        html.H1('Dashboard'),\n        dcc.Graph(figure=px.line(data, x='date', y='value'))\n    ])\n    return app",
        "from bokeh.plotting import figure, show\np = figure(title='Line Plot', x_axis_label='x', y_axis_label='y')\np.line([1,2,3,4], [1,4,9,16], line_width=2)\nshow(p)",
        "def plot_histogram(data, bins=20):\n    plt.figure(figsize=(8, 5))\n    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)\n    plt.axvline(np.mean(data), color='red', linestyle='--', label=f'Mean={np.mean(data):.2f}')\n    plt.legend()",
        "import altair as alt\nchart = alt.Chart(df).mark_circle().encode(\n    x='x:Q', y='y:Q', color='category:N', size='count:Q'\n).interactive()",
    ],
    "math": [
        # Arithmetic (10)
        "1 + 1 = 2",
        "2 * 3 = 6",
        "10 - 4 = 6",
        "15 / 3 = 5",
        "7 + 8 = 15",
        "9 * 9 = 81",
        "100 / 4 = 25",
        "3 + 5 = 8",
        "6 * 7 = 42",
        "20 - 12 = 8",
        # Algebra (10)
        "x^2 + y^2 = r^2",
        "f(x) = ax + b",
        "a^2 + b^2 = c^2 for right triangles",
        "log(a*b) = log(a) + log(b)",
        "(a + b)^2 = a^2 + 2ab + b^2",
        "a^2 - b^2 = (a+b)(a-b)",
        "quadratic formula: x = (-b +- sqrt(b^2-4ac)) / 2a",
        "slope of line: m = (y2-y1)/(x2-x1)",
        "distance formula: d = sqrt((x2-x1)^2 + (y2-y1)^2)",
        "midpoint: M = ((x1+x2)/2, (y1+y2)/2)",
        # Calculus (10)
        "integral of x dx = x^2/2 + C",
        "derivative of x^2 is 2x",
        "derivative of sin(x) is cos(x)",
        "derivative of e^x is e^x",
        "derivative of ln(x) is 1/x",
        "integral of 1/x dx = ln|x| + C",
        "integral of cos(x) dx = sin(x) + C",
        "integral of e^x dx = e^x + C",
        "limit of 1/x as x -> inf is 0",
        "Taylor series: f(x) = sum of f^(n)(a)/n! * (x-a)^n",
        # Trigonometry (8)
        "sin(0) = 0, cos(0) = 1",
        "sin(30) = 0.5, cos(60) = 0.5",
        "sin(90) = 1, cos(90) = 0",
        "tan(45) = 1",
        "sin^2(x) + cos^2(x) = 1",
        "sin(a+b) = sin(a)cos(b) + cos(a)sin(b)",
        "cos(a+b) = cos(a)cos(b) - sin(a)sin(b)",
        "law of cosines: c^2 = a^2 + b^2 - 2ab*cos(C)",
        # Famous formulas (8)
        "e^(i*pi) + 1 = 0",
        "sum of 1 to n is n*(n+1)/2",
        "The area of circle is pi * r^2",
        "The volume of sphere is 4/3 * pi * r^3",
        "The circumference of circle is 2 * pi * r",
        "F = ma (Newton's second law)",
        "E = mc^2 (Einstein's mass-energy equivalence)",
        "Pythagorean theorem: a^2 + b^2 = c^2",
        # Linear Algebra (6)
        "Matrix multiplication is associative",
        "The determinant of 2x2 matrix [[a,b],[c,d]] is ad - bc",
        "Identity matrix has 1s on diagonal and 0s elsewhere",
        "Transpose of matrix A is A^T where (A^T)_ij = A_ji",
        "The inverse of A satisfies A * A^(-1) = I",
        "Eigenvalue equation: Av = lambda*v",
        # Statistics (10)
        "The mean of dataset is sum divided by count of values.",
        "The median is the middle value when data is sorted.",
        "The mode is the value that appears most frequently.",
        "Standard deviation measures the spread of data around mean.",
        "Variance is the average of squared deviations from mean: sigma^2 = sum(x_i-mu)^2 / n.",
        "The normal distribution is bell-shaped and symmetric: N(mu, sigma^2).",
        "Correlation coefficient r ranges from -1 to 1 for linear relationship.",
        "The p-value indicates the probability of observing data under null hypothesis.",
        "Confidence interval gives a range of plausible values for a parameter.",
        "Linear regression models relationship: y = beta_0 + beta_1*x + epsilon.",
        # Probability (10)
        "P(A) = number of favorable outcomes / total possible outcomes.",
        "Expected value E(X) = sum of x_i * P(X=x_i) for discrete random variables.",
        "Bayes theorem: P(A|B) = P(B|A)*P(A) / P(B).",
        "Two events are independent if P(A and B) = P(A) * P(B).",
        "The binomial distribution counts successes in n trials: B(n, p).",
        "Law of large numbers: sample mean converges to expected value as n increases.",
        "Conditional probability: P(A|B) = P(A and B) / P(B).",
        "The sum of probabilities of all possible outcomes equals 1.",
        "Markov chains have the property that future depends only on present state.",
        "Central limit theorem: sampling distribution of mean approaches normal as n grows.",
        # Linear Algebra Pt2 (8)
        "A matrix is singular if and only if its determinant is zero.",
        "The rank of a matrix is the number of linearly independent rows or columns.",
        "Orthogonal vectors have dot product equal to zero.",
        "Gram-Schmidt process orthonormalizes a set of vectors.",
        "Singular value decomposition factors A = U * Sigma * V^T.",
        "The trace of a matrix is the sum of its diagonal elements.",
        "A positive definite matrix has all positive eigenvalues.",
        "Kernel of linear transformation is set of vectors mapped to zero.",
        # Geometry (10)
        "Area of triangle = 1/2 * base * height.",
        "Perimeter of rectangle = 2 * (length + width).",
        "Volume of cylinder = pi * r^2 * h.",
        "The sum of interior angles of triangle is 180 degrees.",
        "Surface area of sphere = 4 * pi * r^2.",
        "Volume of cone = 1/3 * pi * r^2 * h.",
        "Area of trapezoid = 1/2 * (base1 + base2) * height.",
        "The diagonal of rectangle = sqrt(length^2 + width^2).",
        "Volume of rectangular prism = length * width * height.",
        "Arc length of circle sector = (theta/360) * 2 * pi * r.",
        # Number Theory (8)
        "A prime number has exactly two positive divisors: 1 and itself.",
        "The greatest common divisor can be found using Euclidean algorithm.",
        "Modular arithmetic: a ≡ b (mod n) means n divides (a - b).",
        "Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p.",
        "Euler's totient function phi(n) counts integers coprime to n.",
        "The fundamental theorem of arithmetic: every integer has unique prime factorization.",
        "Perfect numbers equal the sum of their proper divisors.",
        "Diophantine equations seek integer solutions to polynomial equations.",
        # Combinatorics (8)
        "Permutation P(n,r) = n! / (n-r)! counts ordered selections of r items from n.",
        "Combination C(n,r) = n! / (r!*(n-r)!) counts unordered selections of r items.",
        "The pigeonhole principle: if n items go into m boxes and n > m, one box has at least 2.",
        "The number of subsets of a set with n elements is 2^n.",
        "The binomial theorem: (x + y)^n = sum of C(n,k)*x^(n-k)*y^k.",
        "Stars and bars: number of solutions to x1+x2+...+xk=n is C(n+k-1, k-1).",
        "Derangements count permutations with no fixed points: !n = n! * sum((-1)^k/k!).",
        "The inclusion-exclusion principle handles overlapping sets: |A∪B| = |A|+|B|-|A∩B|.",
        # Complex Analysis (6)
        "Complex number z = a + bi where i^2 = -1.",
        "Euler's formula: e^(i*theta) = cos(theta) + i*sin(theta).",
        "The modulus of complex number: |z| = sqrt(a^2 + b^2).",
        "Complex conjugate of a+bi is a-bi: z * conj(z) = |z|^2.",
        "De Moivre's theorem: (cos x + i*sin x)^n = cos(nx) + i*sin(nx).",
        "The complex plane maps real part to x-axis and imaginary to y-axis.",
        # Set Theory (5)
        "The empty set is a subset of every set.",
        "The power set P(A) is the set of all subsets of A: |P(A)| = 2^|A|.",
        "Union of sets A and B is A ∪ B = {x | x ∈ A or x ∈ B}.",
        "Intersection of sets A and B is A ∩ B = {x | x ∈ A and x ∈ B}.",
        "Cardinality |A| is the number of elements in set A.",
        # Logic (8)
        "Modus ponens: if P implies Q and P is true, then Q is true.",
        "Contrapositive: P implies Q is equivalent to not Q implies not P.",
        "A tautology is a statement that is always true by its logical form.",
        "Proof by contradiction assumes negation and derives a contradiction.",
        "De Morgan's laws: not(A and B) = not A or not B.",
        "Universal quantifier: ∀x means 'for all x' in the domain.",
        "Existential quantifier: ∃x means 'there exists an x' such that.",
        "Mathematical induction proves P(n) for all n by base case and inductive step.",
        # Calculus Deep (10)
        "Chain rule: derivative of f(g(x)) = f'(g(x)) * g'(x).",
        "Product rule: derivative of u*v = u'*v + u*v'.",
        "Quotient rule: derivative of u/v = (u'*v - u*v') / v^2.",
        "Fundamental theorem of calculus connects differentiation and integration.",
        "Integration by parts: integral of u dv = uv - integral of v du.",
        "L'Hopital's rule: limit of f/g equals limit of f'/g' when 0/0 or inf/inf.",
        "Partial derivatives compute rate of change in one variable holding others constant.",
        "Gradient is the vector of partial derivatives: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z).",
        "Double integrals compute volume under a surface: ∫∫f(x,y)dxdy.",
        "Line integrals integrate a function along a curve in space.",
        # Differential Equations (8)
        "Separable ODE: dy/dx = g(x)*h(y) can be solved by separating variables.",
        "First-order linear ODE: dy/dx + P(x)y = Q(x) solved with integrating factor.",
        "Second-order linear homogeneous: ay'' + by' + cy = 0, solve characteristic ar^2 + br + c = 0.",
        "Euler method approximates ODE solutions: y_(n+1) = y_n + h*f(x_n, y_n).",
        "Laplace transform converts differential equations to algebraic equations.",
        "Partial differential equations model heat, wave, and diffusion phenomena.",
        "The wave equation: ∂^2u/∂t^2 = c^2 * ∂^2u/∂x^2 models vibrating strings.",
        "Fourier series represents periodic functions as sum of sines and cosines.",
        # Group Theory (8)
        "A group (G, *) has closure, associativity, identity, and inverses.",
        "Cyclic groups are generated by a single element: G = <g> = {g^n | n ∈ Z}.",
        "Lagrange's theorem: order of subgroup divides order of group.",
        "Homomorphism preserves group structure: f(a*b) = f(a)*f(b).",
        "Isomorphism is a bijective homomorphism, groups have same structure.",
        "Normal subgroups are invariant under conjugation: gNg^(-1) = N.",
        "Quotient group G/N has cosets as elements with operation induced from G.",
        "Symmetric group S_n consists of all permutations of n elements.",
        # Topology (6)
        "A topological space is a set with a collection of open sets.",
        "Continuous function: preimage of every open set is open.",
        "Compactness: every open cover has a finite subcover.",
        "Connectedness: space cannot be partitioned into two disjoint open sets.",
        "Homeomorphism is a bijective continuous function with continuous inverse.",
        "Metric spaces (X, d) generalize the notion of distance.",
        # Optimization (8)
        "Gradient descent: x_(n+1) = x_n - alpha * ∇f(x_n) to find minimum.",
        "Lagrange multipliers find extrema subject to constraints: ∇f = λ∇g.",
        "Convex function: f(tx+(1-t)y) ≤ t*f(x)+(1-t)*f(y), any local min is global.",
        "Newton's method: x_(n+1) = x_n - f'(x_n)/f''(x_n) for faster convergence.",
        "Linear programming optimizes linear objective with linear constraints.",
        "Kuhn-Tucker conditions generalize Lagrange multipliers for inequalities.",
        "Simplex method solves linear programs by moving along edges of feasible region.",
        "Conjugate gradient method efficiently solves large sparse linear systems Ax=b.",
        # Graph Theory (8)
        "A tree is a connected acyclic graph with n-1 edges for n vertices.",
        "Euler path visits every edge exactly once: exists if at most 2 odd-degree vertices.",
        "Hamiltonian path visits every vertex exactly once: NP-complete to decide.",
        "Planar graphs can be drawn without edge crossings: K5 and K3,3 are not planar.",
        "Bipartite graph: vertices partition into two sets, edges only between sets.",
        "Coloring: chromatic number is minimum colors to color vertices, adjacent differ.",
        "Matching pairs vertices so no vertex appears twice: max matching is polynomial.",
        "Network flow models transport through capacitated directed graph: max-flow min-cut.",
        # Information Theory (6)
        "Entropy H(X) = -sum(p(x)*log2(p(x))) measures uncertainty in bits.",
        "Mutual information I(X;Y) = H(X) - H(X|Y) measures shared information.",
        "Kullback-Leibler divergence D_KL(P||Q) = sum(P(x)*log(P(x)/Q(x))) measures distribution distance.",
        "Channel capacity C = max I(X;Y) over input distributions.",
        "Huffman coding assigns shorter codes to more frequent symbols.",
        "Shannon's source coding theorem: optimal code length approaches entropy.",
        # Numerical Methods (8)
        "Bisection method: repeatedly halve interval where sign changes to find root.",
        "Newton-Raphson: x_(n+1) = x_n - f(x_n)/f'(x_n) for root finding.",
        "Trapezoidal rule approximates integral: h/2 * (f0 + 2f1 + 2f2 + ... + fn).",
        "Gaussian elimination solves linear systems by row operations to upper triangular.",
        "LU decomposition factors matrix A = L*U for efficient solving of Ax=b.",
        "Jacobi iteration: x^(k+1)_i = (b_i - sum_{j≠i} a_ij*x^(k)_j) / a_ii.",
        "Runge-Kutta methods give higher-order accuracy for ODE integration.",
        "Finite difference approximates derivatives: f'(x) ≈ (f(x+h)-f(x))/h.",
        # Abstract Algebra (6)
        "A ring (R, +, *) has addition forming abelian group and multiplication associative.",
        "A field is a ring where nonzero elements form a group under multiplication.",
        "Polynomial ring R[x] consists of polynomials with coefficients in ring R.",
        "Ideals are subsets closed under addition and multiplication by ring elements.",
        "Module generalizes vector space: scalars come from a ring not necessarily a field.",
        "Galois theory connects field extensions with group theory for polynomial solvability.",
        # Number Systems (5)
        "Natural numbers N = {0, 1, 2, ...} are the counting numbers.",
        "Integers Z extend naturals with negative numbers: {..., -2, -1, 0, 1, 2, ...}.",
        "Rational numbers Q = {a/b | a,b ∈ Z, b ≠ 0} are fractions of integers.",
        "Real numbers R complete the rationals by filling all gaps on the number line.",
        "Quaternions H = a + bi + cj + dk with i^2=j^2=k^2=ijk=-1 extend complex numbers.",
        # Applied Math (8)
        "Fourier transform decomposes function into frequency components: F(ω) = ∫f(t)e^(-iωt)dt.",
        "Convolution (f*g)(t) = ∫f(τ)g(t-τ)dτ blends two functions together.",
        "Principal component analysis finds directions of maximum variance in data.",
        "Markov processes model systems transitioning between states probabilistically.",
        "Monte Carlo methods use random sampling to approximate numerical results.",
        "Game theory analyzes strategic interactions between rational decision makers.",
        "Control theory designs systems to achieve desired behavior via feedback loops.",
        "Chaos theory studies systems highly sensitive to initial conditions.",
        # Discrete Math (8)
        "Recurrence relation defines sequence terms from previous ones: a_n = f(a_{n-1}, ...).",
        "Generating functions encode sequences as coefficients of power series.",
        "Boolean algebra operates on truth values with AND, OR, NOT operations.",
        "Finite state machines model computation with states and transitions.",
        "Regular expressions describe patterns in strings using formal rules.",
        "Turing machines are abstract models of computation with infinite tape.",
        "Big-O notation classifies algorithm growth: O(n), O(n log n), O(n^2), O(2^n).",
        "NP-complete problems are hardest in NP: if one solved, all NP easy.",
        # Mathematical Logic (5)
        "Propositional logic deals with statements that are true or false.",
        "Predicate logic extends propositional with quantifiers and predicates.",
        "Godel's incompleteness: any consistent system has statements it cannot prove.",
        "Zermelo-Fraenkel set theory with Choice (ZFC) is standard foundation of math.",
        "Model theory studies mathematical structures satisfying given axioms.",
        # Mathematical Physics (5)
        "Maxwell's equations describe electromagnetic fields: ∇·E = ρ/ε0.",
        "Schrodinger equation: iℏ∂ψ/∂t = Hψ governs quantum state evolution.",
        "Navier-Stokes equations model fluid flow: ρ(∂v/∂t + v·∇v) = -∇p + μ∇^2v.",
        "Noether's theorem: every continuous symmetry corresponds to a conservation law.",
        "Lagrangian mechanics: L = T - V, action S = ∫L dt, principle of least action.",
        # Mathematical Proofs (8)
        "Direct proof: assume P, derive Q through logical steps.",
        "Proof by induction: base case P(1), inductive step P(k) → P(k+1).",
        "Proof by contradiction: assume ¬P, derive contradiction, conclude P.",
        "Proof by contrapositive: prove ¬Q → ¬P to establish P → Q.",
        "Proof by exhaustion: verify all cases individually for finite domain.",
        "Proof by construction: exhibit explicit example satisfying the theorem.",
        "Proof by pigeonhole principle: n items into m < n boxes, one box has at least 2.",
        "Proof by infinite descent: no infinite descending chain of positive integers.",
        # Cryptography (7)
        "RSA encryption: public key (n,e), private key d where e*d ≡ 1 mod φ(n).",
        "Diffie-Hellman key exchange allows two parties to establish shared secret.",
        "AES (Advanced Encryption Standard) uses 128-bit blocks with 128/192/256-bit keys.",
        "SHA-256 produces a 256-bit hash digest from arbitrary input data.",
        "Digital signatures use private key to sign, public key to verify authenticity.",
        "Elliptic curve cryptography provides equivalent security with smaller keys than RSA.",
        "Zero-knowledge proofs allow proving knowledge without revealing the secret.",
        # Mathematical Statistics (7)
        "Maximum likelihood estimation finds parameters maximizing P(data|params).",
        "Hypothesis testing: null H0 vs alternative H1, reject H0 if p-value < alpha.",
        "Type I error: rejecting true H0 (false positive). Type II: accepting false H0.",
        "ANOVA tests whether means of multiple groups are significantly different.",
        "Chi-squared test evaluates independence between categorical variables.",
        "Bootstrap resamples with replacement to estimate sampling distributions.",
        "Bayesian inference updates prior beliefs with observed data: P(θ|data) ∝ P(data|θ)P(θ).",
        # Dynamical Systems (5)
        "Fixed point x* satisfies f(x*) = x* for discrete dynamical system.",
        "Stable equilibrium: nearby initial conditions converge to the equilibrium.",
        "Bifurcation occurs when small parameter change causes qualitative change in behavior.",
        "Limit cycle is an isolated closed trajectory in phase space of dynamical system.",
        "Lyapunov function V(x) proves stability: V(x) > 0 for x ≠ x*, dV/dt < 0.",
    ],
}


def compute_diversity(text: str) -> float:
    """计算文本 token 多样性（unique chars / total chars）"""
    if not text:
        return 0.0
    chars = list(text.replace(" ", "").replace("▁", ""))
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def compute_repetition_ratio(text: str) -> float:
    """计算重复率（最高频 char 出现次数 / 总长度）"""
    if not text:
        return 0.0
    chars = list(text.replace(" ", "").replace("▁", ""))
    if not chars:
        return 0.0
    counter = Counter(chars)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(chars)


def main():
    print("=" * 60)
    print("规模化训练数据生成质量验证")
    print("=" * 60)

    # Step 1: 装配 Cortex
    print("\n[Step 1] 装配 Cortex...")
    from taiji.loader import assemble_cortex
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
    )
    assert cortex._shared_embedding is not None
    total_samples = sum(len(v) for v in TRAINING_DATA.values())
    print(f"  ✅ Cortex 装配完成: {len(cortex.neurons)} neurons")
    print(f"  训练数据: {total_samples} 条, 域分布: {', '.join(f'{d}={len(v)}' for d, v in TRAINING_DATA.items())}")

    # Step 2: 训练前基线生成
    print("\n[Step 2] 训练前基线生成...")
    test_prompts = [
        ("今天天气", "zh"),
        ("人工智能", "zh"),
        ("def hello", "code"),
        ("1+1=", "math"),
    ]
    baselines = {}
    for prompt, domain in test_prompts:
        try:
            gen = cortex.generate(prompt, max_tokens=30, domain=domain, temperature=0.8)
            diversity = compute_diversity(gen)
            rep_ratio = compute_repetition_ratio(gen)
            baselines[prompt] = (gen, diversity, rep_ratio)
            print(f"  [{domain}] '{prompt}' → '{gen[:50]}' (diversity={diversity:.2f}, rep={rep_ratio:.2f})")
        except Exception as e:
            baselines[prompt] = ("", 0.0, 1.0)
            print(f"  [{domain}] '{prompt}' → 生成失败: {e}")

    # Step 3: 多轮 feed+sleep 训练
    print("\n[Step 3] 多轮 feed+sleep 训练...")
    from taiji.life.feed_engine import get_feed_engine
    from taiji.life.sleep_engine import get_sleep_engine, SleepReport

    feed_engine = get_feed_engine()
    sleep_engine = get_sleep_engine()
    sleep_engine.cortex = cortex
    if sleep_engine._feed_engine is None:
        sleep_engine._feed_engine = feed_engine

    NUM_CYCLES = 24  # 覆盖 ~75% 训练数据（24×32=768/987）
    losses_by_cycle = []
    domain_losses_history = {d: [] for d in TRAINING_DATA.keys()}

    for cycle in range(NUM_CYCLES):
        # 每轮喂入全部数据
        for domain, texts in TRAINING_DATA.items():
            for text in texts:
                feed_engine.feed_text(text=text, source=f"cycle_{cycle}", domain=domain)

        # 触发 sleep 训练
        report = SleepReport(timestamp=datetime.now().isoformat(), duration_seconds=0.0)
        sleep_engine._sleep_phase_model_training(report)

        loss = report.training_loss
        losses_by_cycle.append(loss)
        n_samples = report.training_samples_used
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"

        # 打印调质状态（自主进化监控）
        nm = sleep_engine._neuromodulator
        if nm is not None:
            lr_mult = nm.get_lr_multiplier()
            print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n_samples} | "
                  f"DA={nm.dopamine:.2f} 5HT={nm.serotonin:.2f} lr_mult={lr_mult:.2f}")
        else:
            print(f"  Cycle {cycle+1}/{NUM_CYCLES}: loss={loss_str}, samples={n_samples}")

    # Step 4: Loss 趋势分析
    print("\n[Step 4] Loss 趋势分析...")
    valid_losses = [l for l in losses_by_cycle if l is not None]
    if len(valid_losses) >= 2:
        first_loss = valid_losses[0]
        last_loss = valid_losses[-1]
        delta = first_loss - last_loss
        pct = (delta / first_loss * 100) if first_loss > 0 else 0
        print(f"  首轮 loss: {first_loss:.4f}")
        print(f"  末轮 loss: {last_loss:.4f}")
        print(f"  下降量: {delta:.4f} ({pct:+.1f}%)")
        if delta > 0:
            print("  ✅ Loss 持续下降")
        else:
            print("  ⚠️ Loss 未下降")

    # Step 5: 训练后生成对比
    print("\n[Step 5] 训练后生成对比...")
    improvements = 0
    total_compared = 0
    for prompt, domain in test_prompts:
        try:
            gen = cortex.generate(prompt, max_tokens=30, domain=domain, temperature=0.8)
            diversity = compute_diversity(gen)
            rep_ratio = compute_repetition_ratio(gen)
            base_gen, base_div, base_rep = baselines[prompt]

            div_improved = diversity > base_div
            rep_improved = rep_ratio < base_rep
            if div_improved or rep_improved:
                improvements += 1
            total_compared += 1

            div_arrow = "↑" if div_improved else "↓" if diversity < base_div else "="
            rep_arrow = "↓" if rep_improved else "↑" if rep_ratio > base_rep else "="
            print(f"  [{domain}] '{prompt}'")
            print(f"    前: '{base_gen[:40]}' (div={base_div:.2f}, rep={base_rep:.2f})")
            print(f"    后: '{gen[:40]}' (div={diversity:.2f}{div_arrow}, rep={rep_ratio:.2f}{rep_arrow})")
        except Exception as e:
            print(f"  [{domain}] '{prompt}' → 生成失败: {e}")

    # Step 6: Next-token 预测准确率
    print("\n[Step 6] Next-token 预测准确率...")
    coverage_results = {}
    for domain in ["zh"]:
        correct = 0
        total = 0
        top5_hits = 0
        domain_sp = cortex._tokenizer_hub.get_tokenizer(domain)

        for text in TRAINING_DATA[domain][:10]:  # 检查前 10 条
            domain_ids = cortex._tokenizer_hub.encode(text, domain=domain)
            if len(domain_ids) < 4:
                continue

            # 用逐 token 映射构造输入（与训练路径一致）
            general_ids = []
            for did in domain_ids:
                piece = domain_sp.id_to_piece(did)
                gen_ids = cortex._general_sp.EncodeAsIds(piece)
                if gen_ids:
                    general_ids.append(gen_ids[0])

            # 对每个位置，用前缀预测下一个 token
            for i in range(1, min(len(general_ids) - 1, 10)):
                prefix = general_ids[:i+1]
                if len(prefix) < 2:
                    continue
                ids_tensor = torch.tensor([prefix], dtype=torch.long)
                shared_emb = cortex._shared_embedding(ids_tensor)

                with torch.no_grad():
                    result = cortex.think(shared_emb)
                logits = result.get("neuron_logits", {}).get(domain)
                if logits is None:
                    continue

                last_logits = logits[0, -1, :]
                pred_token = torch.argmax(last_logits).item()
                true_token = domain_ids[i+1] if i+1 < len(domain_ids) else domain_ids[-1]

                total += 1
                if pred_token == true_token:
                    correct += 1
                top5 = torch.topk(last_logits, 5).indices.tolist()
                if true_token in top5:
                    top5_hits += 1

        accuracy = correct / total if total > 0 else 0
        top5_acc = top5_hits / total if total > 0 else 0
        coverage_results[domain] = accuracy
        print(f"  [{domain}] next-token 准确率: {accuracy:.1%} ({correct}/{total})")
        print(f"  [{domain}] top-5 准确率: {top5_acc:.1%} ({top5_hits}/{total})")

    # Step 7: 综合判断
    print("\n" + "=" * 60)
    loss_success = len(valid_losses) >= 2 and (valid_losses[0] - valid_losses[-1]) > 0
    quality_success = improvements >= total_compared * 0.5 if total_compared > 0 else False
    coverage_success = any(c > 0.05 for c in coverage_results.values()) if coverage_results else False

    if loss_success and (quality_success or coverage_success):
        print("🎉 验证通过：规模化训练数据改善生成质量")
        print(f"   - Loss: {valid_losses[0]:.4f} → {valid_losses[-1]:.4f}")
        print(f"   - 生成质量改善: {improvements}/{total_compared}")
        if coverage_results:
            print(f"   - Token 覆盖率: {', '.join(f'{d}={c:.1%}' for d, c in coverage_results.items())}")
        return 0
    else:
        print("⚠️ 验证未完全通过")
        print(f"   - Loss 下降: {loss_success}")
        print(f"   - 质量改善: {improvements}/{total_compared}")
        print(f"   - Token 覆盖: {coverage_success}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
