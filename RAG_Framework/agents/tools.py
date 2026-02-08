"""
Tool definitions for the RAG Framework.
Centralizes all tool schemas to avoid duplication.
"""

from RAG_Framework.core.config import ADVANCED_REASONING


def get_tools_for_standard_generator():
    """
    Returns the tools list for the standard generator.

    """
    return [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Searches user's documents for information to respond to the user's query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_document_by_name",
                    "description": "Retrieve an entire document by its filename. Use when user specifically asks for a particular book, report, or document by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": "The name of the document to retrieve (e.g., 'annual_report.pdf')"
                            }
                        },
                        "required": ["document_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_documents",
                    "description": "List available documents in the system. Use when user asks what documents are available. Can filter by keyword - if user asks about specific topics (e.g., 'documents about allometric'), use the filter_keyword parameter to search document names.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_keyword": {
                                "type": "string",
                                "description": "Optional keyword to filter document names (e.g., 'allometric', 'climate', 'policy'). Leave empty to list all documents."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Searches Wikipedia for encyclopedic information on a given topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string for Wikipedia."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "google_custom_search",
                    "description": "Search the internet using Google. Use for current events or general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": "Search the web using DuckDuckGo. Use for current events or general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url_content",
                    "description": "Fetch and extract the main text content from a web page. Use after duckduckgo_search or google_custom_search to get full webpage content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The full URL to fetch"},
                            "max_chars": {"type": "integer", "description": "Max characters to return (default 5000)", "default": 5000}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Execute a read-only SQL SELECT query on a configured database. Only SELECT queries are allowed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": "The name of the database to query (use list_databases to see available options)"
                            },
                            "sql_query": {
                                "type": "string",
                                "description": "The SQL SELECT query to execute. Only SELECT statements are allowed."
                            }
                        },
                        "required": ["db_name", "sql_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_databases",
                    "description": "List all SQL databases available for querying.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_database_schema",
                    "description": "Get the schema (tables and columns) of a database. Use this to understand the database structure before writing queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": "The name of the database to get the schema for"
                            }
                        },
                        "required": ["db_name"]
                    }
                }
            }
        ]


def get_tools_for_agentic_generator():
    """
    Returns the tools list for the agentic reasoning generator.
    """
    return [
        {
                "type": "function",
                "function": {
                    "name": "agentic_generator",
                    "description": "Call advanced task agent if user is asking for a difficult task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Searches user's documents for text chunks relevant to the user's query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_document_by_name",
                    "description": "Retrieve an entire document by its filename. Use when user specifically asks for a particular book, report, or document by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": "The name of the document to retrieve (e.g., 'annual_report.pdf')"
                            }
                        },
                        "required": ["document_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_documents",
                    "description": "List available documents in the system. Use when user asks what documents are available. Can filter by keyword - if user asks about specific topics (e.g., 'documents about allometric'), use the filter_keyword parameter to search document names.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_keyword": {
                                "type": "string",
                                "description": "Optional keyword to filter document names (e.g., 'allometric', 'climate', 'policy'). Leave empty to list all documents."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Search Wikipedia for factual information and general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string for Wikipedia."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "google_custom_search",
                    "description": "Search the internet using Google Custom Search JSON API. Use for current events or information not found in documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": "Search the web using DuckDuckGo. Returns titles, snippets, and URLs. Use fetch_url_content to get full page content from promising results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url_content",
                    "description": "Fetch and extract the main text content from a web page. Use after duckduckgo_search or google_custom_search to get full article content instead of just snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The full URL to fetch"},
                            "max_chars": {"type": "integer", "description": "Max characters to return (default 5000)", "default": 5000}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Execute a read-only SQL SELECT query on a configured database. Only SELECT queries are allowed. Use list_databases to see available databases and get_database_schema to understand the table structure before querying.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": "The name of the database to query (use list_databases to see available options)"
                            },
                            "sql_query": {
                                "type": "string",
                                "description": "The SQL SELECT query to execute. Only SELECT statements are allowed."
                            }
                        },
                        "required": ["db_name", "sql_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_databases",
                    "description": "List all SQL databases available for querying.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_database_schema",
                    "description": "Get the schema (tables and columns) of a database. Use this to understand the database structure before writing queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": "The name of the database to get the schema for"
                            }
                        },
                        "required": ["db_name"]
                    }
                }
            }
    ]
