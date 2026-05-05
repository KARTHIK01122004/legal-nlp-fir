"""
Query Vector Database
=====================

Unified interface for querying the vector database.
Supports complex queries, filtering, and result formatting.
"""

import logging
from typing import List, Dict, Optional, Union
from datetime import datetime
from .ipc_vectordb_builder import IPCVectorDBBuilder
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class VectorDBQuery:
    """Unified query interface for vector database."""
    
    def __init__(
        self,
        embedding_backend: str = "huggingface",
        vector_backend: str = "faiss",
        index_path: str = "./vector_indexes",
    ):
        """
        Initialize query interface.
        
        Args:
            embedding_backend: Embedding backend to use
            vector_backend: Vector storage backend
            index_path: Path to vector indexes
        """
        self.embedding_backend = embedding_backend
        self.vector_backend = vector_backend
        self.index_path = index_path
        
        # Initialize builders/stores
        self.ipc_db = IPCVectorDBBuilder(
            embedding_backend=embedding_backend,
            vector_backend=vector_backend,
            index_path=index_path,
        )
        
        self.embedding_generator = self.ipc_db.embedding_generator
        self.query_history = []
        
        logger.info("Initialized VectorDBQuery interface")
    
    def search_ipc(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        include_details: bool = False,
    ) -> List[Dict]:
        """
        Search IPC sections.
        
        Args:
            query: Search query text
            top_k: Number of results
            threshold: Minimum similarity threshold
            include_details: Include full section details
            
        Returns:
            List of matching IPC sections
        """
        logger.info(f"Searching IPC: {query}")
        
        # Record query
        self.query_history.append({
            "type": "ipc_search",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "top_k": top_k,
        })
        
        # Search
        results = self.ipc_db.search_section(
            query=query,
            top_k=top_k,
            similarity_threshold=threshold,
        )
        
        # Optionally add full details
        if include_details:
            for result in results:
                full_section = self.ipc_db.get_section_details(result["section"])
                if full_section:
                    result["details"] = full_section
        
        return results
    
    def search_by_crime_type(
        self,
        crime_description: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search IPC sections by crime type description.
        
        Args:
            crime_description: Description of the crime
            top_k: Number of results
            
        Returns:
            List of applicable IPC sections
        """
        logger.info(f"Searching by crime type: {crime_description}")
        
        return self.search_ipc(
            query=crime_description,
            top_k=top_k,
            include_details=True,
        )
    
    def search_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search IPC sections by keywords.
        
        Args:
            keywords: List of keywords to search
            top_k: Number of results
            
        Returns:
            List of matching sections
        """
        logger.info(f"Searching by keywords: {keywords}")
        
        self.query_history.append({
            "type": "keyword_search",
            "keywords": keywords,
            "timestamp": datetime.now().isoformat(),
        })
        
        return self.ipc_db.search_by_keywords(keywords, top_k=top_k)
    
    def search_by_category(
        self,
        category: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search IPC sections by category.
        
        Args:
            category: Crime category (e.g., "Sexual Offence", "Theft")
            top_k: Number of results
            
        Returns:
            List of sections in category
        """
        logger.info(f"Searching by category: {category}")
        
        self.query_history.append({
            "type": "category_search",
            "category": category,
            "timestamp": datetime.now().isoformat(),
        })
        
        return self.ipc_db.search_by_category(category, top_k=top_k)
    
    def multi_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Perform multi-faceted search with filtering.
        
        Args:
            query: Search query
            filters: Optional filters (category, punishment_min, etc.)
            top_k: Number of results
            
        Returns:
            Dictionary with semantic and keyword search results
        """
        logger.info(f"Multi-search: {query} with filters: {filters}")
        
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "semantic_results": [],
            "filtered_results": [],
        }
        
        # Semantic search
        semantic_results = self.search_ipc(query, top_k=top_k * 2)
        results["semantic_results"] = semantic_results
        
        # Apply filters if provided
        if filters:
            filtered_results = semantic_results
            
            # Filter by category
            if "category" in filters:
                category_filter = filters["category"].lower()
                filtered_results = [
                    r for r in filtered_results
                    if category_filter in r.get("category", "").lower()
                ]
            
            # Filter by minimum similarity
            if "min_similarity" in filters:
                min_sim = filters["min_similarity"]
                filtered_results = [
                    r for r in filtered_results
                    if r.get("similarity", 0) >= min_sim
                ]
            
            results["filtered_results"] = filtered_results[:top_k]
        else:
            results["filtered_results"] = semantic_results[:top_k]
        
        self.query_history.append({
            "type": "multi_search",
            "query": query,
            "filters": filters,
            "timestamp": datetime.now().isoformat(),
        })
        
        return results
    
    def get_section(self, section_number: str) -> Optional[Dict]:
        """
        Get a specific IPC section by number.
        
        Args:
            section_number: Section number (e.g., "379")
            
        Returns:
            Section details or None
        """
        logger.info(f"Retrieving section: {section_number}")
        
        self.query_history.append({
            "type": "section_lookup",
            "section": section_number,
            "timestamp": datetime.now().isoformat(),
        })
        
        return self.ipc_db.get_section_details(section_number)
    
    def get_related_sections(
        self,
        section_number: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Get sections related to a specific section.
        
        Args:
            section_number: Section number
            top_k: Number of related sections
            
        Returns:
            List of related sections
        """
        logger.info(f"Finding related sections for {section_number}")
        
        self.query_history.append({
            "type": "related_lookup",
            "section": section_number,
            "timestamp": datetime.now().isoformat(),
        })
        
        return self.ipc_db.get_related_sections(section_number, top_k=top_k)
    
    def smart_search(self, complaint_text: str) -> Dict:
        """
        Intelligent search analyzing complaint text.
        
        Args:
            complaint_text: Full complaint text
            
        Returns:
            Comprehensive analysis with recommendations
        """
        logger.info("Performing smart search...")
        
        # Extract key terms (simple keyword extraction)
        words = complaint_text.lower().split()
        keywords = [w for w in words if len(w) > 4][:5]
        
        analysis = {
            "complaint_summary": complaint_text[:200] + "..." if len(complaint_text) > 200 else complaint_text,
            "extracted_keywords": keywords,
            "semantic_results": [],
            "category_results": {},
            "recommendations": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Semantic search
        semantic_results = self.search_ipc(complaint_text, top_k=3)
        analysis["semantic_results"] = semantic_results
        
        # Extract categories from results
        categories = set()
        for result in semantic_results:
            if result.get("category"):
                categories.add(result["category"])
        
        # Search by extracted categories
        for category in categories:
            category_results = self.search_by_category(category, top_k=2)
            analysis["category_results"][category] = category_results
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        self.query_history.append({
            "type": "smart_search",
            "complaint_length": len(complaint_text),
            "timestamp": datetime.now().isoformat(),
        })
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on search analysis."""
        recommendations = []
        
        if analysis["semantic_results"]:
            top_result = analysis["semantic_results"][0]
            section = top_result.get("section")
            title = top_result.get("title")
            recommendations.append(
                f"Primary Applicable Section: IPC {section} - {title}"
            )
            
            if top_result.get("similarity", 0) < 0.5:
                recommendations.append("Low confidence match - Manual review recommended")
        
        if len(analysis["semantic_results"]) > 1:
            recommendations.append(
                f"Multiple applicable sections found - Consider Section "
                f"{analysis['semantic_results'][1].get('section')}"
            )
        
        if analysis["category_results"]:
            recommendations.append(
                f"Crime categories: {', '.join(analysis['category_results'].keys())}"
            )
        
        return recommendations
    
    def compare_sections(
        self,
        section1: str,
        section2: str,
    ) -> Dict:
        """
        Compare two IPC sections.
        
        Args:
            section1: First section number
            section2: Second section number
            
        Returns:
            Comparison analysis
        """
        logger.info(f"Comparing sections {section1} and {section2}")
        
        sec1 = self.get_section(section1)
        sec2 = self.get_section(section2)
        
        if not sec1 or not sec2:
            return {"error": "One or both sections not found"}
        
        comparison = {
            "section1": sec1,
            "section2": sec2,
            "similarities": [],
            "differences": [],
        }
        
        # Compare categories
        if sec1.get("category") == sec2.get("category"):
            comparison["similarities"].append(f"Both in {sec1.get('category')}")
        else:
            comparison["differences"].append(
                f"Different categories: {sec1.get('category')} vs {sec2.get('category')}"
            )
        
        # Compare keywords
        kw1 = set(sec1.get("keywords", []))
        kw2 = set(sec2.get("keywords", []))
        common_keywords = kw1 & kw2
        
        if common_keywords:
            comparison["similarities"].append(f"Common keywords: {', '.join(common_keywords)}")
        
        self.query_history.append({
            "type": "comparison",
            "sections": [section1, section2],
            "timestamp": datetime.now().isoformat(),
        })
        
        return comparison
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history."""
        return self.query_history[-limit:]
    
    def clear_query_history(self):
        """Clear query history."""
        self.query_history.clear()
        logger.info("Query history cleared")
    
    def get_database_stats(self) -> Dict:
        """Get vector database statistics."""
        return self.ipc_db.get_statistics()
    
    def export_search_results(
        self,
        results: List[Dict],
        format: str = "json",
    ) -> Union[str, List[Dict]]:
        """
        Export search results in different formats.
        
        Args:
            results: Search results to export
            format: Export format ('json', 'csv', 'text')
            
        Returns:
            Formatted results
        """
        import json
        
        if format == "json":
            return json.dumps(results, indent=2)
        
        elif format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            if results:
                writer = csv.DictWriter(output, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            
            return output.getvalue()
        
        elif format == "text":
            lines = []
            for result in results:
                lines.append(f"Section {result['section']}: {result['title']}")
                lines.append(f"  Category: {result.get('category')}")
                lines.append(f"  Similarity: {result.get('similarity', 'N/A')}")
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            logger.warning(f"Unknown format: {format}")
            return results
    
    def batch_search(self, queries: List[str], top_k: int = 3) -> Dict:
        """
        Perform batch search on multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Results per query
            
        Returns:
            Dictionary with results for each query
        """
        logger.info(f"Batch searching {len(queries)} queries")
        
        batch_results = {}
        for query in queries:
            batch_results[query] = self.search_ipc(query, top_k=top_k)
        
        return batch_results
