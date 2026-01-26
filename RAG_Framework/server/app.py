"""
Flask-SocketIO server for RAG Framework with real-time WebSocket streaming.
Handles query processing, tool call visualization, and advanced reasoning display.
"""
import json
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import traceback

# Global reference to the RAG system (retriever)
rag_system = None
socketio = None


def create_app(retriever, host='localhost', port=5050):
    """
    Create and configure the Flask-SocketIO application.

    Args:
        retriever: The RAG retriever instance
        host: Server host address
        port: Server port

    Returns:
        Tuple of (app, socketio) instances
    """
    global rag_system, socketio

    # Store retriever for use in event handlers
    rag_system = retriever

    # Create Flask app
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')

    # Enable CORS
    CORS(app)

    # Configure SocketIO
    socketio = SocketIO(app,
                       cors_allowed_origins="*",
                       async_mode='threading',
                       ping_timeout=60,
                       ping_interval=25)

    # Routes
    @app.route('/')
    def index():
        """Serve the main chat interface."""
        return render_template('standalone.html')

    @app.route('/status')
    def status():
        """Health check endpoint."""
        return {'status': 'ready', 'mode': 'websocket'}

    # WebSocket event handlers
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print(f"Client connected: {request.sid if 'request' in dir() else 'unknown'}")
        emit('status', {'state': 'connected', 'message': 'Connected to RAG server'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print(f"Client disconnected - any ongoing generation will continue but won't emit events")

    @socketio.on('abort_generation')
    def handle_abort_generation(data):
        """
        Handle abort generation request.
        Note: MLX generation is blocking and can't be interrupted,
        but we can at least log the abort request.
        The disconnect/reconnect will prevent events from being sent.
        """
        print(f"\n{'!'*60}")
        print(f"Generation abort requested by client")
        print(f"{'!'*60}\n")

    @socketio.on('clear_cache')
    def handle_clear_cache(data):
        """
        Handle cache clearing request from client.
        Clears both the conversation history and the prompt cache.
        """
        print(f"\n{'='*60}")
        print(f"Cache and conversation clear requested")
        print(f"{'='*60}\n")

        try:
            import gc
            from mlx_lm.models.cache import make_prompt_cache

            # Reset conversation history
            if hasattr(rag_system, 'conversation_manager'):
                rag_system.conversation_manager.clear()
                print("Conversation history cleared")

            # Reset prompt cache
            if hasattr(rag_system, 'prompt_cache') and hasattr(rag_system, 'llm_model'):
                # Delete the old cache to free GPU memory
                old_cache = rag_system.prompt_cache
                rag_system.prompt_cache = make_prompt_cache(rag_system.llm_model)
                del old_cache

                # Force garbage collection to clean up GPU resources
                gc.collect()
                print("Prompt cache recreated")

            emit('status', {'state': 'connected', 'message': 'Conversation cleared - Ready for new conversation'})
        except Exception as e:
            print(f"Error clearing cache/conversation: {e}")
            # Try to at least clear conversation
            if hasattr(rag_system, 'conversation_manager'):
                rag_system.conversation_manager.clear()
            emit('status', {'state': 'connected', 'message': 'Ready'})

    @socketio.on('query')
    def handle_query(data):
        """
        Handle incoming query from client.
        Processes query with RAG system and streams results via WebSocket events.

        Expected data format: {'query': str, 'message_id': str}
        """
        query = data.get('query', '')
        message_id = data.get('message_id', 'unknown')

        if not query:
            emit('error', {'message': 'No query provided'})
            return

        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"Message ID: {message_id}")
        print(f"{'='*60}\n")

        # Create streaming callback for this specific query
        def stream_callback(event_type, data):
            """
            Callback function for streaming events from the generator.
            Emits WebSocket events to the connected client.

            Supported event types:
            - 'text_chunk': Token streaming
            - 'tool_call_start': Tool execution begins
            - 'tool_call_result': Tool execution completes
            - 'reasoning_step': Advanced reasoning step update
            - 'reasoning_goal': Goal status update
            - 'reasoning_evaluation': Evaluation results
            - 'status': Status message
            """
            try:
                socketio.emit(event_type, data)
            except Exception as e:
                print(f"Error emitting {event_type}: {e}")

        try:
            # Import here to avoid circular dependency
            from RAG_Framework.components.generators import Generator
            from RAG_Framework.core.config import ADVANCED_REASONING

            # Emit thinking status
            emit('status', {'state': 'thinking', 'message': 'Processing your query...'})

            # Process query with RAG system using appropriate generator
            # Pass the stream_callback to enable real-time streaming
            conversation_manager = rag_system.conversation_manager if hasattr(rag_system, 'conversation_manager') else None
            prompt_cache = rag_system.prompt_cache if hasattr(rag_system, 'prompt_cache') else None

            if ADVANCED_REASONING:
                from RAG_Framework.components.generators.reasoning import AgenticGenerator
                response = AgenticGenerator.agentic_answer_query(
                    query=query,
                    llm_model=rag_system.llm_model,
                    llm_tokenizer=rag_system.llm_tokenizer,
                    retriever=rag_system,
                    prompt_cache=prompt_cache,
                    conversation_manager=conversation_manager
                )
            else:
                response = Generator.answer_query_with_llm(
                    query=query,
                    llm_model=rag_system.llm_model,
                    llm_tokenizer=rag_system.llm_tokenizer,
                    retriever=rag_system,
                    prompt_cache=prompt_cache,
                    stream_callback=stream_callback,
                    verbose=False,
                    conversation_manager=conversation_manager
                )

            # Handle response (could be tuple or string)
            if isinstance(response, tuple):
                response_text, _ = response
            else:
                response_text = response

            # Emit completion
            emit('done', {'message_id': message_id})
            print(f"Query processed successfully. Message ID: {message_id}")

        except Exception as e:
            # Emit error to client
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n{'!'*60}")
            print(f"ERROR: {error_msg}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print(f"{'!'*60}\n")
            emit('error', {'message': error_msg})
            emit('done', {'message_id': message_id, 'error': True})

    return app, socketio


def run_server(retriever, host='localhost', port=5050):
    """
    Create and run the Flask-SocketIO server.

    Args:
        retriever: The RAG retriever instance
        host: Server host address
        port: Server port
    """
    app, sock = create_app(retriever, host, port)

    print(f"\n{'='*60}")
    print(f"Starting RAG WebSocket server")
    print(f"URL: http://{host}:{port}")
    print(f"WebSocket enabled: Yes")
    print(f"{'='*60}\n")

    # Run with SocketIO
    sock.run(app, host=host, port=port, debug=False, use_reloader=False)
