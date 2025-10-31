/**
 * @NApiVersion 2.1
 * @NScriptType Suitelet
 * @NModuleScope SameAccount
 * @author Tanwa Sripan
 *
 * Description:
 * This Suitelet implements a Retrieval-Augmented Generation (RAG) chatbot interface within NetSuite.
 * It uses the N/llm module to provide AI-powered responses enhanced with context from NetSuite records.
 *
 * Key Features:
 * - Interactive chat interface with real-time responses
 * - Vector similarity search for relevant context retrieval
 * - Integration with OCI Cohere's language models
 * - Document ingestion capability for knowledge base building
 * - Custom record storage for embeddings and content
 *
 * The implementation demonstrates:
 * 1. How to use NetSuite's N/llm module for embeddings and chat
 * 2. Vector similarity search implementation
 * 3. RAG pattern with custom knowledge sources
 *
 * MIT License
 *
 * Copyright (c) 2025 Tanwa Sripan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

define(['N/ui/serverWidget', 'N/llm', 'N/query', 'N/runtime', 'N/record'], function (serverWidget, llm, query, runtime, record) {
  function onRequest(context) {
    const isIngestion = context.request.parameters.action === 'ingest';
    if (context.request.method === 'GET') {
      if (isIngestion) {
        displayDataSourceInputInterface(context);
      } else {
        displayChatInterface(context);
      }
    } else {
      if (isIngestion) {
        handleFileIngest(context);
      } else {
        switch(context.request.parameters.step) {
          case 'retrieve':
            handleRetrieveStep(context);
            break;
          case 'augment':
            handleAugmentStep(context);
            break;
          case 'generate':
            handleGenerateStep(context);
            break;
          default:
            handleChatRequest(context);
        }
      }
    }
    log.audit('onRequest remaining usage:', {
      chatUsage: llm.getRemainingFreeUsage(),
      embedUsage: llm.getRemainingFreeEmbedUsage(),
      scriptUsage: runtime.getCurrentScript().getRemainingUsage(),
    });
  }

  /**
   * Display the chat interface that user can interact with
   */
  function displayChatInterface(context) {
    context.response.write({ output: getChatInterfaceHTML() });
  }

  /**
   * Display the data source input interface. User can add text file and said file will be ingested to become
   * knowledge source for RAG. Stored in a custom record.
   */
  function displayDataSourceInputInterface(context) {
    let form = serverWidget.createForm({
      title: 'RAG Data Source Input',
    });

    form.addField({
      id: 'custpage_rag_data_source_file',
      type: serverWidget.FieldType.FILE,
      label: 'Data Source (file must be .txt)',
    });

    const actionField = form.addField({
      id: 'action',
      type: serverWidget.FieldType.TEXT,
      label: 'ingest',
    });
    actionField.defaultValue = 'ingest';
    actionField.updateDisplayType({
      displayType: serverWidget.FieldDisplayType.HIDDEN
    });

    form.addSubmitButton({ label: 'Submit' });

    context.response.writePage(form);
  }

  /**
   * Handle chat requests (POST). Function will perform query embedding and RAG against a pre-config knowledge base
   */
  function handleChatRequest(context) {
    try {
      let params = JSON.parse(context.request.body);
      let userMessage = params.message;
      let conversationHistory = (params.history || []).map((history) => llm.createChatMessage(history));

      log.debug('Chat Request', 'User message: ' + userMessage);

      // Step 1: Retrieval
      let relevantContext = retrieveRelevantContext(userMessage);

      // Step 2 & 3: Augment and Generation
      let llmResponse = generateLLMResponse(userMessage, relevantContext, conversationHistory);

      let response = {
        success: true,
        message: llmResponse.text,
        context: relevantContext,
        model: llmResponse.model,
        timestamp: new Date().toISOString(),
      };

      context.response.write(JSON.stringify(response));
    } catch (e) {
      log.error('handleChatRequest', e.toString());
      context.response.write(
        JSON.stringify({
          success: false,
          error: e.toString(),
        })
      );
    }
  }

  /**
   * Handle file ingest requests (POST). Function will perform text chunking, embed, and store the vector on a custom records
   *
   * llm.embed supports up to 96 inputs
   */
  function handleFileIngest(context) {
    try {
      let file = context.request.files['custpage_rag_data_source_file'];
      if (!file) {
        context.response.write(
          JSON.stringify({
            success: false,
            error: 'No file provided',
          })
        );
        return;
      }

      let fileContent = file.getContents();

      if (!fileContent) {
        context.response.write(
          JSON.stringify({
            success: false,
            error: 'Failed to extract file content',
          })
        );
        return;
      }

      // Chunking files before embedding
      const chunks = simpleTextChunking(fileContent);

      if (!chunks.length) {
        return context.response.write(
          JSON.stringify({
            success: false,
            message: 'Unable to chunk the file',
            stats: {
              filename: file.name,
              total_chunks: chunks.length,
              file_size: file.size,
            },
          })
        );
      }

      // Very limited on governance in Suitelet, may be better to offload to Map Reduce for larger files
      const ingestionDate = new Date().toISOString();
      // llm.embed supports up to 96, each custom record creation cost 6 governance so it is okay for PoC to not consider larger files
      const embeddings = generateEmbedding(chunks.slice(0, 96));
      for (let i = 0; i < embeddings.length; i++) {
        try {
          const metadata = {
            source_file: file.name,
            chunk_index: i,
            total_chunks: chunks.length,
            ingestion_date: ingestionDate,
          };
          const kbRecord = record.create({
            type: 'customrecord_knowledge_base',
          });
          kbRecord
            .setValue('custrecord_kb_embedding', JSON.stringify(embeddings[i]))
            .setValue('custrecord_kb_content', chunks[i])
            .setValue('custrecord_kb_metadata', JSON.stringify(metadata))
            .save();
        } catch (e) {
          log.error('Knowledge base creation error', e.toString());
        }
      }
      // Return ingestion results
      context.response.write(
        JSON.stringify({
          success: true,
          message: 'File ingestion complete',
          stats: {
            filename: file.name,
            total_chunks: chunks.length,
            file_size: file.size,
          },
        })
      );
    } catch (e) {
      log.error('handleFileIngest', e.toString());
      context.response.write(
        JSON.stringify({
          success: false,
          error: e.toString(),
        })
      );
    }
  }

  /**
   * Chunk text into smaller pieces for embedding
   * Uses simple fixed size chunking with overlap
   *
   * Considering ending without cutting off words
   *
   * llm.embed has limit of 512 token, 
   */
  function simpleTextChunking(text, chunkSize = 1500, overlap = 300) {
    chunkSize = chunkSize > 1500 ? 1500 : chunkSize;
    overlap = overlap > 300 ? 300 : overlap;
    if (overlap > chunkSize * 0.2) {
      log.error('simpleTextChunking', 'overlap cannot be larger than 20% of chunk size.');
      return [];
    }
    let chunks = [];

    if (!text || text.length === 0) {
      return chunks;
    }

    let position = 0;

    while (position < text.length) {
      let end = position + chunkSize;
      let chunk = text.slice(position, end);

      // If not at the end, try to break at last space to avoid cutting words
      if (end < text.length) {
        let lastSpace = chunk.lastIndexOf(' ');
        if (lastSpace > 0) {
          chunk = chunk.slice(0, lastSpace);
          end = position + lastSpace;
        }
      }

      chunks.push(chunk.trim());

      // Move position forward by (chunkSize - overlap)
      position = end - overlap;
    }

    return chunks;
  }

  /**
   * Retrieve relevant context using vector similarity search
   * Assumes a custom record 'customrecord_knowledge_base' exists with fields:
   * - custrecord_kb_content (text): The actual content
   * - custrecord_kb_embedding (text): JSON array of embedding vector
   * - custrecord_kb_metadata (text): JSON metadata about the content
   */
  function retrieveRelevantContext(query) {
    let contexts = [];

    try {
      // Step 1: Generate embedding for the user query
      let queryEmbedding = generateEmbedding([query])[0];

      if (!queryEmbedding) {
        log.error('retrieveRelevantContext - Embedding Error', 'Failed to generate query embedding');
        return contexts;
      }

      // Step 2: Retrieve all knowledge base records with embeddings
      let kbRecords = fetchKnowledgeBaseRecords();

      // Step 3: Calculate cosine similarity for each record
      let similarities = [];

      kbRecords.forEach(function (kbRecord) {
        try {
          let storedEmbedding = JSON.parse(kbRecord.embedding);
          let similarity = cosineSimilarity(queryEmbedding, storedEmbedding);
          log.debug("similarity", similarity)
          similarities.push({
            id: kbRecord.id,
            content: kbRecord.content,
            metadata: kbRecord.metadata,
            similarity: similarity,
          });
        } catch (e) {
          log.error('retrieveRelevantContext - Similarity Calculation Error', 'Record ID: ' + kbRecord.id + ', Error: ' + e.toString());
        }
      });

      // Step 4: Sort by similarity (descending) and get top 5 results
      similarities.sort(function (a, b) {
        return b.similarity - a.similarity;
      });

      let topResults = similarities.slice(0, 5);

      // Step 5: Filter results with similarity > 0.7 (threshold for relevance)
      topResults.forEach(function (result) {
        if (result.similarity > 0.3) {
          contexts.push({
            type: 'knowledge_base',
            id: result.id,
            content: result.content,
            metadata: result.metadata ? JSON.parse(result.metadata) : {},
            similarity: result.similarity.toFixed(4),
          });
        }
      });

      log.debug('Context Retrieved', contexts.length + ' relevant items found');
    } catch (e) {
      log.error('retrieveRelevantContext - Context Retrieval Error', e.toString());
    }

    return contexts;
  }

  /**
   * Generate embedding vector for text using N/llm module
   */
  function generateEmbedding(arrText) {
    try {
      let embeddingResponse = llm.embed({
        embedModelFamily: llm.EmbedModelFamily.COHERE_EMBED_ENGLISH,
        inputs: arrText,
      });
      log.debug('embeddingResponse.embeddings[0]', embeddingResponse.embeddings[0]);
      return embeddingResponse.embeddings;
    } catch (e) {
      log.error('generateEmbedding', e.toString());
      return null;
    }
  }

  /**
   * Fetch knowledge base records from custom record
   */
  function fetchKnowledgeBaseRecords() {
    let records = [];

    try {
      // Using SuiteQL to fetch records
      let suiteQLQuery = `
                SELECT 
                    id,
                    custrecord_kb_content as content,
                    custrecord_kb_embedding as embedding,
                    custrecord_kb_metadata as metadata
                FROM 
                    customrecord_knowledge_base
                WHERE 
                    isinactive = 'F'
                    AND custrecord_kb_embedding IS NOT NULL
            `;

      let queryResults = query.runSuiteQL({
        query: suiteQLQuery,
      });

      records = queryResults.asMappedResults().map((result) => ({
        id: result.id,
        content: result.content,
        embedding: result.embedding,
        metadata: result.metadata,
      }));
    } catch (e) {
      log.error('fetchKnowledgeBaseRecords', e.toString());
    }

    return records;
  }

  /**
   * Calculate cosine similarity between two vectors
   * Cosine similarity = (A · B) / (||A|| * ||B||)
   * Returns value between -1 and 1, where 1 means identical vectors pointing in the same direction.
   * Ref: https://en.wikipedia.org/wiki/Cosine_similarity
   */
  function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) {
      return 0;
    }

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i]; // A_i * B_i, from i = 1 to n (0 index for computer)
      magnitudeA += vecA[i] * vecA[i]; // A_i**2, from i = 1 to n (0 index for computer)
      magnitudeB += vecB[i] * vecB[i]; // B_i**2, from i = 1 to n (0 index for computer)
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Generate LLM response using OCI Cohere model via N/llm module
   */
  function generateLLMResponse(userMessage, context, history) {
    try {
      // Augmenting with retrieved data
      let contextStr = 'Relevant NetSuite Data:\n';
      if (context.length > 0) {
        context.forEach(function (ctx) {
          contextStr += JSON.stringify(ctx) + '\n';
        });
      } else {
        contextStr += 'No specific records found for this query.\n';
      }

      // Build messages array for the LLM
      let messages = [];

      // System message with RAG context augmented.
      messages.push(
        llm.createChatMessage({
          role: llm.ChatRole.CHATBOT,
          text: 'You are a helpful NetSuite assistant. Answer questions based on the provided NetSuite data.\n\n' +
             'IMPORTANT: Format your responses for readability:\n' +
             '- Use short paragraphs (2-3 sentences max)\n' +
             '- Add blank lines between paragraphs\n' +
             '- Use bullet points for lists\n' +
             '- Use numbered lists for steps\n' +
             '- Keep responses concise and well-structured\n\n' +
             'Context:\n' + contextStr
        })
      );

      // Add conversation history (limit to last 10 messages for context window)
      let recentHistory = history.slice(-10);
      recentHistory.forEach(function (msg) {
        messages.push(msg);
      });

      // Finally add user's latest query
      messages.push(
        llm.createChatMessage({
          role: llm.ChatRole.USER,
          text: userMessage,
        })
      );

      // Call N/llm module with OCI Cohere model
      let response = llm.chat({
        modelFamily: llm.ModelFamily.COHERE_COMMAND_R_PLUS,
        prompt: userMessage,
        chatHistory: messages,
        modelParameters: {
          maxTokens: 500, // Can play around with this for different response size
          temperature: 0.4, // Can play with this for more "creative" response
        },
      });
      log.debug('response', response);

      return response;
    } catch (e) {
      log.error('generateLLMResponse', e.toString());
      return {
        content: 'Error generating response: ' + e.toString(),
        model: 'Error',
      };
    }
  }

  /**
   * Generate minimal HTML, CSS, vanilla JS for chat interface
   */
  function getChatInterfaceHTML() {
    return `
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1 {
            margin-bottom: 20px;
        }

        .layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }

        .panel {
            background: white;
            border: 1px solid #ddd;
            padding: 15px;
        }

        .panel h3 {
            margin-top: 0;
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
        }

        /* Chat Area */
        #chatMessages {
            height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 10px;
            background: #fafafa;
        }

        .message {
            margin-bottom: 15px;
            padding: 8px;
            border-left: 3px solid #666;
        }

        .message-user {
            background: #e3f2fd;
            border-left-color: #2196F3;
        }

        .message-bot {
            background: #f5f5f5;
            border-left-color: #666;
        }

        .message-label {
            font-weight: bold;
            font-size: 0.85em;
            margin-bottom: 3px;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        #userInput {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            font-size: 14px;
        }

        button {
            padding: 8px 20px;
            background: #2196F3;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }

        button:hover {
            background: #1976D2;
        }

        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        /* RAG Process */
        .rag-steps {
            list-style: none;
            padding: 0;
        }

        .rag-steps li {
            padding: 8px;
            margin-bottom: 8px;
            border: 1px solid #ddd;
            background: #f9f9f9;
        }

        .rag-steps li.active {
            background: #2196F3;
            color: white;
            border-color: #2196F3;
        }

        /* Context Display */
        #contextDisplay {
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.9em;
        }

        .context-item {
            padding: 8px;
            margin-bottom: 8px;
            background: #f9f9f9;
            border: 1px solid #ddd;
            font-family: monospace;
            font-size: 0.85em;
        }

        .context-type {
            font-weight: bold;
            color: #2196F3;
        }

        .context-item pre {
            white-space: pre-wrap;
            word-break: break-word; 
            overflow-wrap: break-word;
        }

        /* Stats */
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }

        .stat-box {
            text-align: center;
            padding: 10px;
            border: 1px solid #ddd;
        }

        .stat-value {
            font-size: 24px;
            font-weight: bold;
        }

        .stat-label {
            font-size: 0.85em;
            color: #666;
        }

        .loading {
            display: none;
            padding: 10px;
            background: #fff3cd;
            border: 1px solid #ffc107;
            margin-bottom: 10px;
        }

        .loading.active {
            display: block;
        }

        .model-info {
            font-size: 0.85em;
            color: #666;
            margin-top: 10px;
            padding: 8px;
            background: #f0f0f0;
            border-left: 3px solid #2196F3;
        }

        /* Thinking Bubble Animation */
        .thinking-bubble {
            display: none;
            margin: 10px 0;
            padding: 15px;
            background: #e3f2fd;
            position: relative;
        }

        .thinking-bubble.active {
            display: block;
        }

        .thinking-dots {
            display: flex;
            gap: 4px;
            padding: 3px;
        }

        .thinking-dots span {
            width: 8px;
            height: 8px;
            background: #2196F3;
            border-radius: 50%;
            animation: thinking 1.4s infinite;
            opacity: 0.5;
        }

        .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes thinking {
            0%, 100% { opacity: 0.5; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>NetSuite RAG Chatbot</h1>
        <div class="model-info">
            Model: OCI Cohere Command R+ | Context: NetSuite Custom Record
        </div>

        <div class="layout">
            <!-- Chat Panel -->
            <div class="panel">
                <h3>Chat</h3>
                <div class="loading" id="loading">Processing request...</div>
                <div id="chatMessages">
                    <div class="message message-bot">
                        <div class="message-label">Assistant</div>
                        <div>Hello! I'm your NetSuite RAG assistant powered by OCI Cohere. Ask me about your integration documentations.</div>
                    </div>
                </div>
                <div class="thinking-bubble" id="thinkingBubble">
                    <div class="thinking-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
                <div class="input-group">
                    <input 
                        type="text" 
                        id="userInput" 
                        placeholder="Ask a question..."
                        onkeypress="if(event.key==='Enter') sendMessage()"
                    />
                    <button onclick="sendMessage()" id="sendBtn">Send</button>
                </div>
            </div>

            <!-- Info Panel -->
            <div>
                <div class="panel">
                    <h3>RAG Process</h3>
                    <ol class="rag-steps">
                        <li id="step1">1. Receive Query</li>
                        <li id="step2">2. Retrieve Context</li>
                        <li id="step3">3. Augment Prompt</li>
                        <li id="step4">4. Generating Response</li>
                    </ol>
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h3>Retrieved Context</h3>
                    <div id="contextDisplay">
                        <p style="color: #999;">No context retrieved yet</p>
                    </div>
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h3>Statistics</h3>
                    <div class="stats">
                        <div class="stat-box">
                            <div class="stat-value" id="messageCount">0</div>
                            <div class="stat-label">Messages</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="contextCount">0</div>
                            <div class="stat-label">Context Items</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let conversationHistory = [];
        let messageCount = 0;
        let currentContext = null;
        let currentAugmentedPrompt = null;

        async function executeStep(step, data) {
            const endpoint = window.location.href + '&step=' + step;
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                return await response.json();
            } catch (error) {
                throw new Error('Step ' + step + ' failed: ' + error);
            }
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;

            // Disable UI
            input.disabled = true;
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('loading').classList.add('active');

            try {
                // Show thinking bubble during LLM generation
                document.getElementById('thinkingBubble').classList.add('active');
                
                // Step 1: Query received
                addMessage(message, 'user');
                document.getElementById('step1').classList.add('active');
                // Step 2: Retrieval
                const retrieveResult = await executeStep('retrieve', { message });
                document.getElementById('step2').classList.add('active');
                if (!retrieveResult.success) throw new Error(retrieveResult.error);
                currentContext = retrieveResult.context;
                updateContext(currentContext);

                // Step 3: Augmenting
                const augmentResult = await executeStep('augment', { 
                    message,
                    context: currentContext 
                });
                if (!augmentResult.success) throw new Error(augmentResult.error);
                document.getElementById('step3').classList.add('active');
                currentAugmentedPrompt = augmentResult.augmentedPrompt;

                // Step 4: Generation
                document.getElementById('step4').classList.add('active');
                const generateResult = await executeStep('generate', {
                    message,
                    context: currentContext,
                    augmentedPrompt: currentAugmentedPrompt,
                    history: conversationHistory
                });
                if (!generateResult.success) throw new Error(generateResult.error);

                // Update UI with results
                addMessage(generateResult.response.text, 'bot');
                
                // Update conversation history
                conversationHistory.push(
                    { role: 'USER', text: message },
                    { role: 'CHATBOT', text: generateResult.response.text }
                );

            } catch (error) {
                addMessage('Error: ' + error.toString(), 'bot');
            } finally {
                // Hide thinking bubble
                document.getElementById('thinkingBubble').classList.remove('active');
                document.getElementById('loading').classList.remove('active');
                resetSteps();
                input.disabled = false;
                document.getElementById('sendBtn').disabled = false;
                input.value = '';
                input.focus();
            }
        }

        function addMessage(content, type) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message message-' + type;
            
            const label = document.createElement('div');
            label.className = 'message-label';
            label.textContent = type === 'user' ? 'You' : 'Assistant';
            
            const contentDiv = document.createElement('div');

            if (type !== 'user') {
                // Convert double newlines to paragraphs, single newlines to <br>
                var formatted = content
                    .split('\\n\\n')
                    .map(function(para) {
                        return '<p style="margin-bottom: 12px;">' + 
                              para.replace(/\\n/g, '<br>') + 
                              '</p>';
                    })
                    .join('');
                contentDiv.innerHTML = formatted;
            } else {
                contentDiv.textContent = content;
            }
            
            messageDiv.appendChild(label);
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            messageCount++;
            document.getElementById('messageCount').textContent = messageCount;
        }

        function updateContext(contexts) {
            const contextDiv = document.getElementById('contextDisplay');
            contextDiv.innerHTML = '';

            if (!contexts || contexts.length === 0) {
                contextDiv.innerHTML = '<p style="color: #999;">No context retrieved</p>';
                document.getElementById('contextCount').textContent = '0';
                return;
            }

            contexts.forEach(ctx => {
                const item = document.createElement('div');
                item.className = 'context-item';
                item.innerHTML = '<span class="context-type">' + ctx.type + '</span><br>' +
                                 '<pre>' + JSON.stringify(ctx, null, 2) + '</pre>';
                contextDiv.appendChild(item);
            });

            document.getElementById('contextCount').textContent = contexts.length;
        }

        function resetSteps() {
            const steps = ['step1', 'step2', 'step3', 'step4'];
            steps.forEach(step => {
                document.getElementById(step).classList.remove('active');
            });
        }
    </script>
</body>
</html>
        `;
  }

  /**
   * Handle retrieve step requests (POST). Function will perform retrieval of relevant context
   */
  function handleRetrieveStep(context) {
    try {
        let params = JSON.parse(context.request.body);
        let userMessage = params.message;
        
        let relevantContext = retrieveRelevantContext(userMessage);
        
        context.response.write(JSON.stringify({
            success: true,
            step: 'retrieve',
            context: relevantContext
        }));
    } catch (e) {
        log.error('handleRetrieveStep', e);
        context.response.write(JSON.stringify({
            success: false,
            step: 'retrieve',
            error: e.toString()
        }));
    }
  }

  /**
   * Handle augment step requests (POST). Planceholder function for the animation of the step
   */
  function handleAugmentStep(context) {
    try {
        // Augmenting is just creating a new prompt from context, so that is done already with the generateLLMResponse        
        context.response.write(JSON.stringify({
            success: true,
            step: 'augment'
        }));
    } catch (e) {
        log.error('handleAugmentStep', e);
        context.response.write(JSON.stringify({
            success: false,
            step: 'augment',
            error: e.toString()
        }));
    }
  }

  /**
   * Handle generate step requests (POST). Function will generate response using augmented prompt
   */
  function handleGenerateStep(context) {
    try {
        let params = JSON.parse(context.request.body);
        let history = (params.history || []).map((history) => llm.createChatMessage(history));
        
        let response = generateLLMResponse(params.message, params.context, history);
        
        context.response.write(JSON.stringify({
            success: true,
            step: 'generate',
            response: response
        }));
    } catch (e) {
        log.error('handleGenerateStep', e);
        context.response.write(JSON.stringify({
            success: false,
            step: 'generate',
            error: e.toString()
        }));
    }
  }

  return {
    onRequest: onRequest,
  };
});
