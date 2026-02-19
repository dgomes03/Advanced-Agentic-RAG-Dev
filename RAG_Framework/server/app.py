"""
Flask-SocketIO server for RAG Framework with real-time WebSocket streaming.
Handles query processing, tool call visualization, and advanced reasoning display.
"""
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pathlib import Path
import traceback

from .chat_storage import get_chat_storage

# Global reference to the RAG system (retriever)
rag_system = None
socketio = None


def create_app(retriever, host='0.0.0.0', port=5050):
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

    # Chat persistence API routes
    @app.route('/api/chats', methods=['GET'])
    def list_chats():
        """List all conversations."""
        storage = get_chat_storage()
        chats = storage.list_chats()
        return jsonify(chats)

    @app.route('/api/chats', methods=['POST'])
    def create_chat():
        """Create a new conversation."""
        storage = get_chat_storage()
        data = request.get_json() or {}
        title = data.get('title')
        chat = storage.create_chat(title=title)
        return jsonify(chat), 201

    @app.route('/api/chats/<chat_id>', methods=['GET'])
    def get_chat(chat_id):
        """Get a specific conversation."""
        storage = get_chat_storage()
        chat = storage.load_chat(chat_id)
        if chat is None:
            return jsonify({'error': 'Chat not found'}), 404
        return jsonify(chat)

    @app.route('/api/chats/<chat_id>', methods=['PUT'])
    def update_chat(chat_id):
        """Update a conversation."""
        storage = get_chat_storage()
        data = request.get_json() or {}
        messages = data.get('messages', [])
        title = data.get('title')
        chat = storage.update_chat(chat_id, messages, title)
        if chat is None:
            return jsonify({'error': 'Chat not found'}), 404
        return jsonify(chat)

    @app.route('/api/chats/<chat_id>', methods=['DELETE'])
    def delete_chat(chat_id):
        """Delete a conversation and its KV-cache file."""
        storage = get_chat_storage()
        if storage.delete_chat(chat_id):
            # Also delete the KV-cache file if it exists
            cache_path = Path(__file__).parent / 'chats' / f"{chat_id}.safetensors"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    print(f"Deleted KV-cache file: {cache_path}")
                except IOError as e:
                    print(f"Failed to delete cache file {cache_path}: {e}")
            return jsonify({'success': True})
        return jsonify({'error': 'Chat not found'}), 404

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

            # Reset tracked chat_id
            rag_system.current_chat_id = None

            emit('status', {'state': 'connected', 'message': 'Conversation cleared - Ready for new conversation'})
        except Exception as e:
            print(f"Error clearing cache/conversation: {e}")
            # Try to at least clear conversation
            if hasattr(rag_system, 'conversation_manager'):
                rag_system.conversation_manager.clear()
            rag_system.current_chat_id = None
            emit('status', {'state': 'connected', 'message': 'Ready'})

    @socketio.on('restore_chat')
    def handle_restore_chat(data):
        """
        Restore a conversation's KV-cache and conversation manager from disk.
        Called when the user loads a previous conversation in the sidebar.
        """
        chat_id = data.get('chat_id')
        messages = data.get('messages', [])

        print(f"\n{'='*60}")
        print(f"Restoring conversation: {chat_id}")
        print(f"{'='*60}\n")

        try:
            import gc
            import json as _json
            from mlx_lm.models.cache import make_prompt_cache, load_prompt_cache

            chat_storage_dir = Path(__file__).parent / 'chats'
            cache_path = chat_storage_dir / f"{chat_id}.safetensors"
            saved_conversation = None

            # Try to load cached KV-cache from disk
            if cache_path.exists():
                try:
                    old_cache = getattr(rag_system, 'prompt_cache', None)
                    loaded_cache, metadata = load_prompt_cache(str(cache_path), return_metadata=True)
                    rag_system.prompt_cache = loaded_cache
                    if old_cache is not None:
                        del old_cache
                    gc.collect()

                    # Extract full conversation state from metadata
                    conv_json = metadata.get("conversation")
                    if conv_json:
                        saved_conversation = _json.loads(conv_json)

                    cache_offset = rag_system.prompt_cache[0].offset if rag_system.prompt_cache else 0
                    print(f"KV-cache loaded from {cache_path} (offset={cache_offset})")

                    # Trim generated response tokens from the cache.
                    # The saved cache has prompt + generated tokens, but only the prompt
                    # tokens form a valid prefix for future prompts.
                    prompt_token_count_str = metadata.get("prompt_token_count")
                    if prompt_token_count_str:
                        prompt_token_count = int(prompt_token_count_str)
                        trim_amount = cache_offset - prompt_token_count
                        if trim_amount > 0:
                            for layer in rag_system.prompt_cache:
                                layer.trim(trim_amount)
                            print(f"Cache trimmed: {cache_offset} -> {prompt_token_count} tokens (removed {trim_amount} generated tokens)")
                except Exception as e:
                    print(f"Failed to load cache file (may be incompatible): {e}")
                    rag_system.prompt_cache = make_prompt_cache(rag_system.llm_model)
                    print("Created fresh prompt cache as fallback")
            else:
                # No cache file — create fresh (first query will prefill)
                old_cache = getattr(rag_system, 'prompt_cache', None)
                rag_system.prompt_cache = make_prompt_cache(rag_system.llm_model)
                if old_cache is not None:
                    del old_cache
                gc.collect()
                print(f"No cache file found, created fresh prompt cache")

            # Restore conversation_manager:
            # Prefer full conversation from cache metadata (includes tool calls/results)
            # so the token sequence matches the saved KV-cache exactly.
            # Fall back to UI messages if metadata is unavailable.
            if hasattr(rag_system, 'conversation_manager'):
                if saved_conversation:
                    rag_system.conversation_manager.load_from_state(saved_conversation)
                    print(f"Conversation restored from cache metadata ({len(saved_conversation)} entries)")
                else:
                    rag_system.conversation_manager.load_from_messages(messages)
                    print(f"Conversation restored from UI messages ({len(messages)} messages, no cache metadata)")

            # Track active chat_id for future cache saves
            rag_system.current_chat_id = chat_id

            emit('status', {'state': 'connected', 'message': 'Conversation restored'})
        except Exception as e:
            print(f"Error restoring conversation: {e}")
            traceback.print_exc()
            # Try to at least restore conversation manager
            if hasattr(rag_system, 'conversation_manager'):
                rag_system.conversation_manager.load_from_messages(messages)
            rag_system.current_chat_id = chat_id
            emit('status', {'state': 'connected', 'message': 'Conversation restored (without cache)'})

    @socketio.on('save_chat')
    def handle_save_chat(data):
        """
        Handle save chat request from client.
        Saves messages to the chat file.
        """
        chat_id = data.get('chat_id')
        messages = data.get('messages', [])
        title = data.get('title')

        storage = get_chat_storage()

        if chat_id:
            # Update existing chat
            chat = storage.update_chat(chat_id, messages, title)
            if chat:
                rag_system.current_chat_id = chat_id
                emit('chat_saved', {'chat_id': chat_id, 'chat': chat})
            else:
                emit('error', {'message': 'Failed to save chat'})
        else:
            # Create new chat
            chat = storage.create_chat(title=title)
            chat = storage.update_chat(chat['id'], messages, title)
            rag_system.current_chat_id = chat['id']
            emit('chat_saved', {'chat_id': chat['id'], 'chat': chat, 'is_new': True})

    @socketio.on('load_chat')
    def handle_load_chat(data):
        """
        Handle load chat request from client.
        Loads chat from storage and sends to client.
        """
        chat_id = data.get('chat_id')
        if not chat_id:
            emit('error', {'message': 'No chat_id provided'})
            return

        storage = get_chat_storage()
        chat = storage.load_chat(chat_id)

        if chat:
            emit('chat_loaded', {'chat_id': chat_id, 'chat': chat})
        else:
            emit('error', {'message': 'Chat not found'})

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

        # Handle exit command - shut down the server
        if query.strip().lower() == 'exit':
            print("\nExit command received from client. Shutting down server...")
            emit('server_shutdown', {'message': 'Server is shutting down...'})
            import threading
            def shutdown():
                import time
                time.sleep(0.5)
                import os, signal
                os.kill(os.getpid(), signal.SIGINT)
            threading.Thread(target=shutdown, daemon=True).start()
            return

        # Handle restart command - restart the entire RAG process
        if query.strip().lower() == 'restart':
            print("\nRestart command received from client. Restarting server...")
            emit('server_restart', {'message': 'Server is restarting...'})
            import threading, sys
            def restart():
                import time, os, stat
                time.sleep(0.5)
                # Close only socket FDs to release the port,
                # leaving regular file FDs intact so Python can exec
                for fd in range(3, 256):
                    try:
                        if stat.S_ISSOCK(os.fstat(fd).st_mode):
                            os.close(fd)
                    except OSError:
                        pass
                os.execv(sys.executable, [sys.executable] + sys.argv)
            threading.Thread(target=restart, daemon=True).start()
            return

        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"Message ID: {message_id}")
        print(f"{'='*60}\n")

        # Capture the requesting client's session so streaming events
        # target it specifically (emit() instead of socketio.emit()).
        sid = request.sid

        def stream_callback(event_type, data):
            """Emit a streaming event to the requesting client."""
            try:
                emit(event_type, data, to=sid)
            except Exception as e:
                print(f"Error emitting {event_type}: {e}")

        try:
            # Import here to avoid circular dependency
            from RAG_Framework.components.generators import Generator
            from RAG_Framework.core.config import ADVANCED_REASONING, REASONING_MODEL

            # Emit thinking status
            emit('status', {'state': 'thinking', 'message': 'Processing your query...'})

            # Process query with RAG system using appropriate generator
            # Pass the stream_callback to enable real-time streaming
            conversation_manager = rag_system.conversation_manager if hasattr(rag_system, 'conversation_manager') else None
            prompt_cache = rag_system.prompt_cache if hasattr(rag_system, 'prompt_cache') else None

            if ADVANCED_REASONING or not REASONING_MODEL:
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
            elif REASONING_MODEL:
                from RAG_Framework.components.generators.LRM import LRMGenerator
                response = LRMGenerator.answer_query_with_llm(
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
                response_text, last_prompt = response
            else:
                response_text = response
                last_prompt = None

            # Emit completion (include response text as fallback for the frontend)
            emit('done', {'message_id': message_id, 'response': response_text})
            print(f"\nResponse complete. (Message ID: {message_id})")

            # Save KV-cache to disk for this conversation
            # Include full conversation state as metadata so the cache
            # can be restored with the exact token sequence it was built from
            chat_id = getattr(rag_system, 'current_chat_id', None)
            if chat_id and prompt_cache:
                try:
                    import json as _json
                    from mlx_lm.models.cache import save_prompt_cache
                    cache_path = str(Path(__file__).parent / 'chats' / f"{chat_id}.safetensors")
                    metadata = {}
                    if conversation_manager:
                        metadata["conversation"] = _json.dumps(
                            conversation_manager.get_conversation(), ensure_ascii=False
                        )
                    # Save prompt token count so generated tokens can be trimmed on restore.
                    # The cache currently has prompt + generated tokens, but only the prompt
                    # tokens form a valid prefix for future prompts (generated tokens don't
                    # match the chat template re-formatted version of the same response).
                    if last_prompt is not None:
                        prompt_token_count = len(rag_system.llm_tokenizer.encode(last_prompt, add_special_tokens=False))
                        metadata["prompt_token_count"] = str(prompt_token_count)
                    save_prompt_cache(cache_path, prompt_cache, metadata)
                    print(f"KV-cache saved to {cache_path}")
                except Exception as e:
                    print(f"Failed to save KV-cache: {e}")

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


def run_server(retriever, host='0.0.0.0', port=5050):
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
