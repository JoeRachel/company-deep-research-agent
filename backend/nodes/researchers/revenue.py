from typing import Any, Dict

from langchain_core.messages import AIMessage

from ...classes import ResearchState
from .base import BaseResearcher


class NewsScanner(BaseResearcher):
    def __init__(self) -> None:
        super().__init__()
        self.analyst_type = "revenue_analyzer"

    async def analyze(self, state: ResearchState) -> Dict[str, Any]:
        company = state.get('company', 'Unknown Company')
        msg = [f"📰 Revenue Scanner analyzing {company}"]
        
        # Generate search queries using LLM
        queries = await self.generate_queries(state, """
        生成用于查找{company}营业收入占比情况的搜索查询，例如：
        - 按业务板块或产品线划分的收入占比
        - 最新财务报告或收益拆解
        """)

        subqueries_msg = "🔍 Subqueries for revenue analysis:\n" + "\n".join([f"• {query}" for query in queries])
        messages = state.get('messages', [])
        messages.append(AIMessage(content=subqueries_msg))
        state['messages'] = messages
        
        revenue_data = {}
        
        # If we have site_scrape data, include it first
        if site_scrape := state.get('site_scrape'):
            msg.append("\n📊 Including site scrape data in company analysis...")
            company_url = state.get('company_url', 'company-website')
            revenue_data[company_url] = {
                'title': state.get('company', 'Unknown Company'),
                'raw_content': site_scrape,
                'query': f'关于{company}的不同类型业务营业收入占比情况'  # Add a default query for site scrape
            }
        
        # Perform additional research with recent time filter
        try:
            # Store documents with their respective queries
            for query in queries:
                documents = await self.search_documents(state, [query])
                if documents:  # Only process if we got results
                    for url, doc in documents.items():
                        doc['query'] = query  # Associate each document with its query
                        revenue_data[url] = doc
            
            msg.append(f"\n✓ Found {len(revenue_data)} documents")
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message=f"Used Tavily Search to find {len(revenue_data)} documents",
                        result={
                            "step": "Searching",
                            "analyst_type": "News Scanner",
                            "queries": queries
                        }
                    )
        except Exception as e:
            msg.append(f"\n⚠️ Error during research: {str(e)}")
        
        # Update state with our findings
        messages = state.get('messages', [])
        messages.append(AIMessage(content="\n".join(msg)))
        state['messages'] = messages
        state['revenue_data'] = revenue_data
        
        return {
            'message': msg,
            'revenue_data': revenue_data
        }

    async def run(self, state: ResearchState) -> Dict[str, Any]:
        return await self.analyze(state) 