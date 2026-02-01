"""RAG (Retrieval-Augmented Generation) tool for searching documentation.

This tool loads Markdown documents from the _docs directory, creates embeddings,
and provides a search interface for retrieving relevant documentation using
LangChain retrievers for better integration and quality.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool, StructuredTool

from agent.tools.base import Tool, ToolContext, ToolResult


class RAGTool(Tool):
    """RAG tool for searching documentation using vector similarity search.
    
    This tool loads Markdown documents from the _docs directory, creates
    embeddings using HuggingFace models, stores them in Chroma, and provides
    a search operation to retrieve relevant documentation chunks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RAG tool.
        
        Args:
            config: Tool configuration containing:
                - docs_path: Path to documentation directory (default: _docs)
                - embedding_model: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)
                - persist_directory: Directory to persist Chroma DB (default: .chroma_db)
                - chunk_size: Text chunk size for splitting (default: 800, optimized for Markdown)
                - chunk_overlap: Overlap between chunks (default: 150)
                - k: Number of documents to retrieve (default: 8)
                - score_threshold: Minimum similarity score threshold (default: 0.75)
                - retriever_type: Type of retriever to use (default: "similarity_score_threshold")
                    Options: "similarity_score_threshold", "similarity", "mmr", "ensemble"
                - use_ensemble: Whether to use EnsembleRetriever with BM25 (default: False)
                    If True, combines BM25 (lexical) + vector (semantic) retrieval
        """
        super().__init__(config)
        self.docs_path = Path(config.get("docs_path", "_docs"))
        self.embedding_model = config.get(
            "embedding_model", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = config.get("persist_directory", ".chroma_db")
        # Optimized defaults for Markdown documentation
        self.chunk_size = config.get("chunk_size", 800)
        self.chunk_overlap = config.get("chunk_overlap", 150)
        self.k = config.get("k", 8)
        self.score_threshold = config.get("score_threshold", 0.75)
        self.retriever_type = config.get("retriever_type", "similarity_score_threshold")
        self.use_ensemble = config.get("use_ensemble", False)
        
        # Initialize components lazily
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[Chroma] = None
        self._retriever: Optional[BaseRetriever] = None
        self._documents: Optional[List[Document]] = None  # Store for BM25 if needed
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Return tool name."""
        return "rag_documentation_search"
    
    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Search and retrieve relevant documentation from the codebase documentation. "
            "Use this tool when you need to find information about LangChain, tools, "
            "agents, or other topics covered in the documentation. "
            "Input should be a search query describing what information you're looking for."
        )
    
    def _ensure_initialized(self) -> None:
        """Initialize embeddings and vector store if not already done."""
        if self._initialized:
            return
        
        # Initialize embeddings
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        
        # Load or create vector store
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(exist_ok=True)
        
        if persist_path.exists() and any(persist_path.iterdir()):
            # Load existing vector store
            try:
                self._vectorstore = Chroma(
                    persist_directory=str(persist_path),
                    embedding_function=self._embeddings
                )
                # Check if it has documents
                if self._vectorstore._collection.count() > 0:
                    # Create retriever from existing vector store
                    self._create_retriever()
                    self._initialized = True
                    return
            except Exception as e:
                print(f"Warning: Failed to load existing vector store: {e}")
                # Will create new one below
        
        # Create new vector store from documents
        self._load_documents()
        self._initialized = True
    
    def _create_retriever(self) -> None:
        """Create retriever from vector store based on configuration.
        
        Supports multiple retriever types:
        - similarity_score_threshold: Best for technical docs, filters by score
        - similarity: Standard similarity search
        - mmr: Maximum Marginal Relevance (diversity + relevance)
        - ensemble: Combines BM25 (lexical) + vector (semantic)
        """
        if self._vectorstore is None:
            raise ValueError("Vector store must be initialized before creating retriever")
        
        if self.use_ensemble or self.retriever_type == "ensemble":
            # Ensemble retriever: BM25 + vector (best for technical documentation)
            try:
                # Import EnsembleRetriever (try langchain_classic first, then langchain)
                try:
                    from langchain_classic.retrievers.ensemble import EnsembleRetriever  # type: ignore
                except ImportError:
                    from langchain.retrievers import EnsembleRetriever  # type: ignore
                
                # Import BM25Retriever from langchain_community
                from langchain_community.retrievers import BM25Retriever  # type: ignore
                
                if self._documents is None:
                    # Need to reload documents for BM25
                    # This is a limitation - we'd need to store chunks separately
                    # For now, fall back to vector-only
                    print("Warning: Documents not available for BM25, using vector-only retriever")
                    self._create_vector_retriever()
                    return
                
                # Create BM25 retriever (lexical search - great for exact terms, code)
                bm25 = BM25Retriever.from_documents(self._documents)
                bm25.k = max(4, self.k // 2)  # BM25 gets half the results
                
                # Create vector retriever (semantic search)
                vector_retriever = self._vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": self.k,
                        "score_threshold": self.score_threshold
                    }
                )
                
                # Combine both retrievers
                self._retriever = EnsembleRetriever(
                    retrievers=[bm25, vector_retriever],
                    weights=[0.4, 0.6]  # 40% BM25, 60% vector (tuned for docs)
                )
                self.retriever_type = "ensemble"
                print("Created EnsembleRetriever (BM25 + Vector)")
                return
            except ImportError:
                print("Warning: EnsembleRetriever not available, falling back to vector-only")
                self._create_vector_retriever()
                return
        
        # Vector-only retrievers
        self._create_vector_retriever()
    
    def _create_vector_retriever(self) -> None:
        """Create vector-based retriever (similarity, threshold, or MMR)."""
        if self.retriever_type == "similarity_score_threshold":
            # Recommended for technical documentation
            self._retriever = self._vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.k,
                    "score_threshold": self.score_threshold
                }
            )
        elif self.retriever_type == "mmr":
            # Maximum Marginal Relevance (diversity + relevance)
            self._retriever = self._vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.k,
                    "fetch_k": min(20, self.k * 3),  # Fetch more, return diverse subset
                    "lambda_mult": 0.7  # Balance: 0=diversity, 1=relevance
                }
            )
        else:
            # Default: standard similarity search
            self._retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.k
                }
            )
    
    def _load_documents(self) -> None:
        """Load Markdown documents from _docs directory and create vector store."""
        # Resolve path relative to project root if relative
        if not self.docs_path.is_absolute():
            # Try to find project root (where this file is located)
            project_root = Path(__file__).parent.parent.parent
            self.docs_path = (project_root / self.docs_path).resolve()
        
        if not self.docs_path.exists():
            raise ValueError(f"Documentation path does not exist: {self.docs_path}")
        
        # Find all Markdown files
        md_files = list(self.docs_path.glob("*.md"))
        if not md_files:
            raise ValueError(f"No Markdown files found in {self.docs_path}")
        
        # Load documents
        documents: List[Document] = []
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(md_file),
                                "filename": md_file.name,
                                "title": md_file.stem
                            }
                        )
                    )
            except Exception as e:
                print(f"Warning: Failed to load {md_file}: {e}")
                continue
        
        if not documents:
            raise ValueError("No documents could be loaded")
        
        # Split documents into chunks with Markdown-aware separators
        # This preserves headers and improves recall for technical documentation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n## ",      # H2 headers
                "\n### ",     # H3 headers
                "\n#### ",    # H4 headers
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                " ",          # Spaces
                ""            # Fallback
            ]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Store documents for potential BM25 retriever (ensemble mode)
        self._documents = chunks
        
        # Create vector store
        persist_path = Path(self.persist_directory)
        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(persist_path)
        )
        
        # Create retriever from vector store
        self._create_retriever()
        
        print(f"Loaded {len(documents)} documents ({len(chunks)} chunks) into vector store")
        print(f"Retriever type: {self.retriever_type}")
    
    async def execute(
        self,
        operation: str,
        arguments: Dict[str, Any],
        ctx: Optional[ToolContext] = None
    ) -> ToolResult:
        """Execute RAG search operation.
        
        Args:
            operation: Operation name (should be "search")
            arguments: Dictionary containing:
                - query: Search query string
                - k: Optional number of results (overrides default)
            ctx: Optional tool context
        
        Returns:
            ToolResult with search results
        """
        if not self.enabled:
            return ToolResult(
                success=False,
                output=None,
                error="RAG tool is disabled"
            )
        
        if operation != "search":
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation: {operation}. Supported: 'search'"
            )
        
        query = arguments.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(
                success=False,
                output=None,
                error="Missing or invalid 'query' argument"
            )
        
        try:
            # Initialize if needed
            self._ensure_initialized()
            
            if self._retriever is None:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Retriever not initialized"
                )
            
            # Get number of results (only applies to non-ensemble retrievers)
            k = arguments.get("k", self.k)
            if not isinstance(k, int) or k <= 0:
                k = self.k
            
            # Use retriever to get relevant documents
            # This is the LangChain-recommended approach
            docs = self._retriever.get_relevant_documents(query)
            
            # Format results
            formatted_results = []
            for doc in docs:
                # For retrievers, we may not have scores directly
                # Try to get score if available (similarity_score_threshold returns it)
                score = getattr(doc, "score", None)
                if score is None:
                    # For ensemble or MMR, we don't have direct scores
                    # Could compute similarity if needed, but it's expensive
                    score = None
                
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score) if score is not None else None
                })
            
            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results)
                },
                metadata={
                    "k": k,
                    "model": self.embedding_model,
                    "retriever_type": self.retriever_type
                }
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"RAG search failed: {str(e)}"
            )


def create_rag_langchain_tool(rag_tool: RAGTool) -> Any:
    """Create a LangChain tool from RAGTool with RAG category.
    
    Args:
        rag_tool: RAGTool instance
    
    Returns:
        LangChain tool categorized as "RAG" (different from browser search tools)
    """
    def rag_documentation_search(query: str, k: int = 4) -> str:
        """Search and retrieve relevant documentation from the codebase.
        
        Use this tool when you need to find information about LangChain, tools,
        agents, or other topics covered in the documentation.
        
        Args:
            query: Search query describing what information you're looking for
            k: Number of document chunks to retrieve (default: 4)
        
        Returns:
            JSON string with search results containing relevant documentation chunks
        """
        import asyncio
        import json
        
        # Execute the tool
        result = asyncio.run(
            rag_tool.execute(
                operation="search",
                arguments={"query": query, "k": k}
            )
        )
        
        if not result.success:
            return json.dumps({"error": result.error})
        
        # Format output for readability
        output = result.output
        formatted = f"Found {output['count']} relevant document(s) for query: '{output['query']}'\n\n"
        
        for i, res in enumerate(output['results'], 1):
            score_str = f"{res['score']:.4f}" if res['score'] is not None else "N/A"
            formatted += f"--- Result {i} (score: {score_str}) ---\n"
            formatted += f"Source: {res['metadata'].get('filename', 'unknown')}\n"
            formatted += f"Content:\n{res['content']}\n\n"
        
        return formatted
    
    # Create tool with RAG category using StructuredTool for explicit category control
    return StructuredTool.from_function(
        func=rag_documentation_search,
        name="rag_documentation_search",
        description=(
            "Search and retrieve relevant documentation from the codebase documentation. "
            "Use this tool when you need to find information about LangChain, tools, "
            "agents, or other topics covered in the documentation. "
            "Input should be a search query describing what information you're looking for."
        ),
        tags=["RAG"],  # Categorized as RAG, separate from browser search tools
    )
