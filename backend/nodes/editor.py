import logging
import os
from typing import Any, Dict

from langchain_core.messages import AIMessage
from openai import AsyncOpenAI

from ..classes import ResearchState
from ..utils.references import format_references_section

logger = logging.getLogger(__name__)




class Editor:
    """Compiles individual section briefings into a cohesive final report."""
    
    def __init__(self) -> None:
        self.siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
        if not self.siliconflow_key:
            raise ValueError("SILICONFLOW_API_KEY environment variable is not set")
        
        # Configure OpenAI
        self.openai_client = AsyncOpenAI(api_key=self.siliconflow_key,base_url="https://api.siliconflow.cn/v1")
        
        # Initialize context dictionary for use across methods
        self.context = {
            "company": "Unknown Company",
            "industry": "Unknown",
            "hq_location": "Unknown"
        }

    async def compile_briefings(self, state: ResearchState) -> ResearchState:
        """Compile individual briefing categories from state into a final report."""
        company = state.get('company', 'Unknown Company')
        
        # Update context with values from state
        self.context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown')
        }
        
        # Send initial compilation status
        if websocket_manager := state.get('websocket_manager'):
            if job_id := state.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="processing",
                    message=f"Starting report compilation for {company}",
                    result={
                        "step": "Editor",
                        "substep": "initialization"
                    }
                )

        context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown')
        }
        
        msg = [f"📑 Compiling final report for {company}..."]
        
        # Pull individual briefings from dedicated state keys
        briefing_keys = {
            'company': 'company_briefing',
            'industry': 'industry_briefing',
            'financial': 'financial_briefing',
            'revenue': 'revenue_briefing'
        }

        # Send briefing collection status
        if websocket_manager := state.get('websocket_manager'):
            if job_id := state.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="processing",
                    message="Collecting section briefings",
                    result={
                        "step": "Editor",
                        "substep": "collecting_briefings"
                    }
                )

        individual_briefings = {}
        for category, key in briefing_keys.items():
            if content := state.get(key):
                individual_briefings[category] = content
                msg.append(f"Found {category} briefing ({len(content)} characters)")
            else:
                msg.append(f"No {category} briefing available")
                logger.error(f"Missing state key: {key}")
        
        if not individual_briefings:
            msg.append("\n⚠️ No briefing sections available to compile")
            logger.error("No briefings found in state")
        else:
            try:
                compiled_report = await self.edit_report(state, individual_briefings, context)
                if not compiled_report or not compiled_report.strip():
                    logger.error("Compiled report is empty!")
                else:
                    logger.info(f"Successfully compiled report with {len(compiled_report)} characters")
            except Exception as e:
                logger.error(f"Error during report compilation: {e}")
        state.setdefault('messages', []).append(AIMessage(content="\n".join(msg)))
        return state
    
    async def edit_report(self, state: ResearchState, briefings: Dict[str, str], context: Dict[str, Any]) -> str:
        """Compile section briefings into a final report and update the state."""
        try:
            company = self.context["company"]
            
            # Step 1: Initial Compilation
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Compiling initial research report",
                        result={
                            "step": "Editor",
                            "substep": "compilation"
                        }
                    )

            edited_report = await self.compile_content(state, briefings, company)
            if not edited_report:
                logger.error("Initial compilation failed")
                return ""

            # Step 2: Deduplication and Cleanup
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Cleaning up and organizing report",
                        result={
                            "step": "Editor",
                            "substep": "cleanup"
                        }
                    )

            # Step 3: Formatting Final Report
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Formatting final report",
                        result={
                            "step": "Editor",
                            "substep": "format"
                        }
                    )
            final_report = await self.content_sweep(state, edited_report, company)
            
            final_report = final_report or ""
            
            logger.info(f"Final report compiled with {len(final_report)} characters")
            if not final_report.strip():
                logger.error("Final report is empty!")
                return ""
            
            logger.info("Final report preview:")
            logger.info(final_report[:500])
            
            # Update state with the final report in two locations
            state['report'] = final_report
            state['status'] = "editor_complete"
            if 'editor' not in state or not isinstance(state['editor'], dict):
                state['editor'] = {}
            state['editor']['report'] = final_report
            logger.info(f"Report length in state: {len(state.get('report', ''))}")
            
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="editor_complete",
                        message="Research report completed",
                        result={
                            "step": "Editor",
                            "report": final_report,
                            "company": company,
                            "is_final": True,
                            "status": "completed"
                        }
                    )
            
            return final_report
        except Exception as e:
            logger.error(f"Error in edit_report: {e}")
            return ""
    
    async def compile_content(self, state: ResearchState, briefings: Dict[str, str], company: str) -> str:
        """Initial compilation of research sections."""
        combined_content = "\n\n".join(content for content in briefings.values())
        
        references = state.get('references', [])
        reference_text = ""
        if references:
            logger.info(f"Found {len(references)} references to add during compilation")
            
            # Get pre-processed reference info from curator
            reference_info = state.get('reference_info', {})
            reference_titles = state.get('reference_titles', {})
            
            logger.info(f"Reference info from state: {reference_info}")
            logger.info(f"Reference titles from state: {reference_titles}")
            
            # Use the references module to format the references section
            reference_text = format_references_section(references, reference_info, reference_titles)
            logger.info(f"Added {len(references)} references during compilation")
        
        # Use values from centralized context
        company = self.context["company"]
        industry = self.context["industry"]
        hq_location = self.context["hq_location"]
        
        prompt = f"""你正在撰写一份关于 {company} 的综合研究报告。

以下是收集到的研究简报内容：
{combined_content}

请基于这些内容，撰写一份完整且重点明确的研究报告。该公司是 {industry} 行业的企业。

写作要求：
1. 融合所有部分内容，形成连贯且无重复的叙述；
2. 每一部分的重要信息都需保留；
3. 内容组织要有逻辑，去除过渡性解释或评论性语言；
4. 使用清晰的章节标题和结构进行排版。

⚠️ 请**严格遵循以下文档结构格式**：

# {company} 研究报告

## 公司概况  
[公司相关内容，可包含若干 ### 小节]

## 行业概况  
[行业相关内容，可包含若干 ### 小节]

## 财务概况  
[财务相关内容，可包含若干 ### 小节]

## 营业收入占比概况   
[营业收入占比概况相关内容，可包含若干 ### 小节]

请以**干净的 Markdown 格式**输出完整报告，**不要添加解释说明或评论文字**。
"""

        
        try:
            response = await self.openai_client.chat.completions.create(
                model="Qwen/Qwen3-235B-A22B",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的报告编辑，擅长将多段研究简报整合为结构清晰、内容全面的公司调研报告。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                stream=False
            )
            initial_report = response.choices[0].message.content.strip()
            
            # Append the references section after LLM processing
            if reference_text:
                initial_report = f"{initial_report}\n\n{reference_text}"
            
            return initial_report
        except Exception as e:
            logger.error(f"Error in initial compilation: {e}")
            return (combined_content or "").strip()
        
    async def content_sweep(self, state: ResearchState, content: str, company: str) -> str:
        """Sweep the content for any redundant information."""
        # Use values from centralized context
        company = self.context["company"]
        industry = self.context["industry"]
        hq_location = self.context["hq_location"]
        
        prompt = f"""你是一位专业的简报编辑专家。你将收到一份关于「{company}」的报告。
当前报告内容：
{content}

请按如下要求处理报告：

1. 删除冗余或重复的信息  
2. 删除与{company}无关的信息  
3. 删除内容空洞或缺乏实质信息的部分  
4. 删除所有元评论（例如“以下是新闻...”之类的句子）

**严格遵守以下文档结构格式：**

## 公司概况  
[包含子标题（使用 ###）的公司相关内容]

## 行业概况  
[包含子标题（使用 ###）的行业相关内容]

## 财务概况  
[包含子标题（使用 ###）的财务相关内容]

## 营业收入占比概况  
[包含子标题（使用 ###）的营业收入占比情况相关内容]

## 参考  
[MLA 格式的参考文献 —— 保持其原样，**不要修改格式**]

**关键规则：**  
1. 文档必须以标题「# {company}研究报告」开头  
2. 文档只允许使用以下这些顺序和格式的二级标题（##）：
   - ## 公司概况  
   - ## 行业概况  
   - ## 财务概况  
   - ## 营业收入占比概况  
   - ## 参考  
3. 不允许使用其他任何二级标题（##）  
4. 在 公司概况 / 行业概况 / 财务概况 / 营业收入占比概况  部分中，只能使用三级标题（###）作为小节标题  
5. 严禁使用代码块（```）  
6. 禁止在各部分之间出现超过一个空行  
7. 所有项目符号统一使用「*」  
8. 每个部分或列表前后必须留一个空行  
9. 严禁更改 参考 部分的格式 
10.请在生成报告中的“参考”，**不要直接显示完整的链接地址（如 https://xxx.com/...）。对于每一个链接，请尝试访问该链接（或根据链接路径推测标题），获取网页的中文标题或最有代表性的名称（如“诺和诺德官网”、“Nai500：诺和诺德股价深度回调”）。
   然后用 Markdown 语法 [标题文本](链接地址) 进行引用。示例：错误示例：https://nai500.com/zh-hans/blog/2025/04/%E8%AF%BA...   正确示例：[诺和诺德股价深度回调后值得买入吗](https://nai500.com/zh-hans/blog/2025/04/诺和诺德...)



返回整理后的报告，要求以**完美的 Markdown 格式输出**，不附加任何解释或说明。
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="Qwen/Qwen3-235B-A22B", 
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位专业的 Markdown 格式化专家，擅长统一文档结构，确保格式清晰、一致、规范。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                stream=True
            )
            
            accumulated_text = ""
            buffer = ""
            
            async for chunk in response:
                if chunk.choices[0].finish_reason == "stop":
                    websocket_manager = state.get('websocket_manager')
                    if websocket_manager and buffer:
                        job_id = state.get('job_id')
                        if job_id:
                            await websocket_manager.send_status_update(
                                job_id=job_id,
                                status="report_chunk",
                                message="Formatting final report",
                                result={
                                    "chunk": buffer,
                                    "step": "Editor"
                                }
                            )
                    break
                    
                chunk_text = chunk.choices[0].delta.content
                if chunk_text:
                    accumulated_text += chunk_text
                    buffer += chunk_text
                    
                    if any(char in buffer for char in ['.', '!', '?', '\n']) and len(buffer) > 10:
                        if websocket_manager := state.get('websocket_manager'):
                            if job_id := state.get('job_id'):
                                await websocket_manager.send_status_update(
                                    job_id=job_id,
                                    status="report_chunk",
                                    message="Formatting final report",
                                    result={
                                        "chunk": buffer,
                                        "step": "Editor"
                                    }
                                )
                        buffer = ""
            
            return (accumulated_text or "").strip()
        except Exception as e:
            logger.error(f"Error in formatting: {e}")
            return (content or "").strip()

    async def run(self, state: ResearchState) -> ResearchState:
        state = await self.compile_briefings(state)
        # Ensure the Editor node's output is stored both top-level and under "editor"
        if 'report' in state:
            if 'editor' not in state or not isinstance(state['editor'], dict):
                state['editor'] = {}
            state['editor']['report'] = state['report']
        return state
