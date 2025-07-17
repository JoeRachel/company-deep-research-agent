import asyncio
import logging
import os
from typing import Any, Dict, List, Union

import google.generativeai as genai
from openai import AsyncOpenAI
from ..classes import ResearchState

logger = logging.getLogger(__name__)

class Briefing:
    """Creates briefings for each research category and updates the ResearchState."""
    
    def __init__(self) -> None:
        self.max_doc_length = 12000  # Maximum document content length
        self.siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.siliconflow_key:
            raise ValueError("SILICONFLOW_API_KEY environment variable is not set")
        
        self.openai_client = AsyncOpenAI(api_key=self.siliconflow_key,base_url="https://api.siliconflow.cn/v1")

    async def generate_category_briefing(
        self, docs: Union[Dict[str, Any], List[Dict[str, Any]]], 
        category: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        company = context.get('company', 'Unknown')
        industry = context.get('industry', 'Unknown')
        hq_location = context.get('hq_location', 'Unknown')
        logger.info(f"Generating {category} briefing for {company} using {len(docs)} documents")

        # Send category start status
        if websocket_manager := context.get('websocket_manager'):
            if job_id := context.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="briefing_start",
                    message=f"Generating {category} briefing",
                    result={
                        "step": "Briefing",
                        "category": category,
                        "total_docs": len(docs)
                    }
                )

        prompts = {
            'company': f"""为{company}公司生成一份简明公司概况，该公司属于{industry}行业。
关键要求：
1. 以以下格式开头：“{company}是一家[什么类型的公司]，其主要业务是[做什么]，面向的客户是[谁]”
2. 使用以下标题和项目符号组织结构，标题必须完全一致：

### 核心产品/服务
* 列出明确的产品或功能
* 仅包含经验证的技术能力

### 管理团队
* 列出主要管理团队成员
* 包括他们的职位和专业背景

### 目标市场
* 列出明确的目标用户群体
* 列出经验证的使用场景
* 列出已确认的客户/合作伙伴

### 核心差异化
* 列出独特的功能或特性
* 列出经过验证的优势

### 商业模式
* 说明产品/服务的定价方式
* 列出销售/分销渠道

3. 每个项目符号必须是一个完整的、可核实的事实
4. 禁止使用“未找到信息”或“暂无数据”等措辞
5. 只允许使用项目符号列表，不允许使用段落
6. 仅提供简报内容，不提供任何解释或评论。""",

    'industry': f"""为{company}公司生成一份聚焦行业的简报，该公司属于{industry}行业。
关键要求：
1. 使用以下标题和项目符号组织结构，标题必须完全一致：

### 市场概况
* 明确{company}所在的市场细分领域
* 给出市场规模及对应年份
* 给出市场增长率及对应年份范围

### 直接竞争对手
* 列出直接竞争对手名称
* 列出具体的竞争产品
* 列出它们在市场上的定位

### 竞争优势
• 列出独特的技术特性
• 列出经过验证的优势

### 市场挑战
• 列出具体、可验证的挑战

2. 每个项目符号都必须是一个明确的、可验证的新闻事件
3. 只允许使用项目符号列表，不允许使用段落
4. 禁止使用“未找到信息”或“暂无数据”等措辞
5. 仅提供简报内容，不提供任何解释。""",

    'financial': f"""为{company}公司生成一份聚焦财务信息的简报，该公司属于{industry}行业。
关键要求：
1. 使用以下标题和项目符号组织结构：

### 融资与投资
* 总融资金额及对应时间
* 每一轮融资信息及时间
* 列出参与的投资者名称

### 收入模式
* 如果适用，请说明产品/服务的定价方式

2. 尽可能包含具体数值
3. 只允许使用项目符号列表，不允许使用段落
4. 禁止使用“未找到信息”或“暂无数据”等措辞
5. 严禁重复记录同一个融资轮次；如果同月发生多轮融资，默认合并为一轮
6. 严禁使用融资金额区间。请根据提供的信息自行判断并给出一个具体数值
7. 仅提供简报内容，不提供任何解释或评论。""",

    'revenue': f"""为{company}公司生成一份聚焦公司不同类型业务收入及其占比的简报，该公司属于{industry}行业。

关键要求：
1. 使用以下标题和项目符号组织结构：

### 收入构成
* 按业务板块或产品线列出具体收入及占总收入的百分比
* 如有可用信息，注明各业务的收入变化趋势（按年或季度）

### 最新财报摘要
* 引用最近一次财务报告中披露的核心收入数据
* 包括总收入及同比增长率
* 可包括地区收入分布（如适用）

2. 尽可能包含具体数值和时间点
3. 只允许使用项目符号列表，不允许使用段落
4. 禁止使用“未找到信息”或“暂无数据”等措辞
5. 严禁使用收入范围。请根据已知信息判断并给出具体数值
6. 仅提供简报内容，不提供任何解释或评论。
""",
}
        
        # Normalize docs to a list of (url, doc) tuples
        items = list(docs.items()) if isinstance(docs, dict) else [
            (doc.get('url', f'doc_{i}'), doc) for i, doc in enumerate(docs)
        ]
        # Sort documents by evaluation score (highest first)
        sorted_items = sorted(
            items, 
            key=lambda x: float(x[1].get('evaluation', {}).get('overall_score', '0')), 
            reverse=True
        )
        
        doc_texts = []
        total_length = 0
        for _ , doc in sorted_items:
            title = doc.get('title', '')
            content = doc.get('raw_content') or doc.get('content', '')
            if len(content) > self.max_doc_length:
                content = content[:self.max_doc_length] + "... [content truncated]"
            doc_entry = f"Title: {title}\n\nContent: {content}"
            if total_length + len(doc_entry) < 140000:  # Keep under limit
                doc_texts.append(doc_entry)
                total_length += len(doc_entry)
            else:
                break
        
        separator = "\n" + "-" * 40 + "\n"
        prompt = f"""{prompts.get(category, f'请根据所提供的文档，为公司 {company} 在 {industry} 行业中的情况撰写一份有重点、具洞察力的研究简报。')} 
请分析以下文档，提取关键信息，仅输出研究简报内容，不要包含解释或评论：
{separator}{separator.join(doc_texts)}{separator}

"""
        
        try:
            logger.info("Sending prompt to LLM")
            response = await self.openai_client.chat.completions.create(
                            model="deepseek-ai/DeepSeek-V3",  # 或其他模型名，比如 "deepseek-v3"
                            messages=[
                                {"role": "user", "content": prompt}
                            ])
            content = response.choices[0].message.content.strip()
            if not content:
                logger.error(f"Empty response from LLM for {category} briefing")
                return {'content': ''}

            # Send completion status
            if websocket_manager := context.get('websocket_manager'):
                if job_id := context.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="briefing_complete",
                        message=f"Completed {category} briefing",
                        result={
                            "step": "Briefing",
                            "category": category
                        }
                    )

            return {'content': content}
        except Exception as e:
            logger.error(f"Error generating {category} briefing: {e}")
            return {'content': ''}

    async def create_briefings(self, state: ResearchState) -> ResearchState:
        """Create briefings for all categories in parallel."""
        company = state.get('company', 'Unknown Company')
        websocket_manager = state.get('websocket_manager')
        job_id = state.get('job_id')
        
        # Send initial briefing status
        if websocket_manager and job_id:
            await websocket_manager.send_status_update(
                job_id=job_id,
                status="processing",
                message="Starting research briefings",
                result={"step": "Briefing"}
            )

        context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown'),
            "websocket_manager": websocket_manager,
            "job_id": job_id
        }
        logger.info(f"Creating section briefings for {company}")
        
        # Mapping of curated data fields to briefing categories
        categories = {
            'company_data': ("company", "company_briefing"),
            'revenue_data': ("revenue", "revenue_briefing"),
            'financial_data': ("financial", "financial_briefing"),
            'industry_data': ("industry", "industry_briefing"),
        }
        
        briefings = {}

        # Create tasks for parallel processing
        briefing_tasks = []
        for data_field, (cat, briefing_key) in categories.items():
            curated_key = f'curated_{data_field}'
            curated_data = state.get(curated_key, {})
            
            if curated_data:
                logger.info(f"Processing {data_field} with {len(curated_data)} documents")
                
                # Create task for this category
                briefing_tasks.append({
                    'category': cat,
                    'briefing_key': briefing_key,
                    'data_field': data_field,
                    'curated_data': curated_data
                })
            else:
                logger.info(f"No data available for {data_field}")
                state[briefing_key] = ""

        # Process briefings in parallel with rate limiting
        if briefing_tasks:
            # Rate limiting semaphore for LLM API
            briefing_semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent briefings
            
            async def process_briefing(task: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single briefing with rate limiting."""
                async with briefing_semaphore:
                    result = await self.generate_category_briefing(
                        task['curated_data'],
                        task['category'],
                        context
                    )
                    
                    if result['content']:
                        briefings[task['category']] = result['content']
                        state[task['briefing_key']] = result['content']
                        logger.info(f"Completed {task['data_field']} briefing ({len(result['content'])} characters)")
                    else:
                        logger.error(f"Failed to generate briefing for {task['data_field']}")
                        state[task['briefing_key']] = ""
                    
                    return {
                        'category': task['category'],
                        'success': bool(result['content']),
                        'length': len(result['content']) if result['content'] else 0
                    }

            # Process all briefings in parallel
            results = await asyncio.gather(*[
                process_briefing(task) 
                for task in briefing_tasks
            ])
            
            # Log completion statistics
            successful_briefings = sum(1 for r in results if r['success'])
            total_length = sum(r['length'] for r in results)
            logger.info(f"Generated {successful_briefings}/{len(briefing_tasks)} briefings with total length {total_length}")

        state['briefings'] = briefings
        return state

    async def run(self, state: ResearchState) -> ResearchState:
        return await self.create_briefings(state)