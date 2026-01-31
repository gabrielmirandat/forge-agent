/**Chat page - main chat interface with sidebar.

Layout similar to ChatGPT:
- Left sidebar with session history
- Main area with chat messages
- Input at bottom
*/

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';

// CSS for animations
const animationStyle = document.createElement('style');
animationStyle.textContent = `
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;
if (!document.head.querySelector('style[data-animations]')) {
  animationStyle.setAttribute('data-animations', 'true');
  document.head.appendChild(animationStyle);
}

// Function to normalize markdown content - remove extra spaces and normalize line breaks
function normalizeMarkdown(content: string): string {
  // Split into lines
  let lines = content.split('\n');
  
  // Remove trailing spaces from each line
  lines = lines.map(line => line.trimEnd());
  
  // Remove multiple consecutive empty lines (keep max 1)
  let normalized: string[] = [];
  let emptyCount = 0;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isEmpty = line.trim() === '';
    
    if (isEmpty) {
      emptyCount++;
      // Keep only one empty line at a time
      if (emptyCount === 1) {
        normalized.push('');
      }
    } else {
      emptyCount = 0;
      // Fix list items - ensure proper spacing after numbers/bullets
      // Pattern: "1. " or "- " at start of line (with optional spaces before)
      const listItemMatch = line.match(/^(\s*)(\d+\.|\-|\*)\s*/);
      if (listItemMatch) {
        // Ensure single space after list marker
        const indent = listItemMatch[1];
        const marker = listItemMatch[2];
        const rest = line.substring(listItemMatch[0].length).trim();
        // Remove extra spaces in the content but preserve structure
        const cleanedRest = rest.replace(/\s+/g, ' ');
        // Only add if there's content after the marker
        if (cleanedRest) {
          normalized.push(`${indent}${marker} ${cleanedRest}`);
        } else {
          // If marker is alone, keep it but ensure proper format
          normalized.push(`${indent}${marker}`);
        }
      } else {
        // Check for lines that start with just a number (broken list items)
        // Pattern: line starts with number followed by space but no period
        const brokenNumberMatch = line.match(/^(\s*)(\d+)\s+(.+)$/);
        if (brokenNumberMatch) {
          // Fix broken list item - add period after number
          const indent = brokenNumberMatch[1];
          const number = brokenNumberMatch[2];
          const content = brokenNumberMatch[3].trim().replace(/\s+/g, ' ');
          normalized.push(`${indent}${number}. ${content}`);
        } else {
          // For non-list lines, normalize multiple spaces to single space
          // But preserve leading spaces (for indentation)
          const leadingSpaces = line.match(/^(\s*)/)?.[1] || '';
          const content = line.trim();
          // Normalize multiple spaces in content
          const normalizedContent = content.replace(/\s+/g, ' ');
          normalized.push(leadingSpaces + normalizedContent);
        }
      }
    }
  }
  
  // Join back
  let result = normalized.join('\n');
  
  // Remove multiple spaces (but preserve in code blocks and list markers)
  // First, protect code blocks
  const codeBlockRegex = /```[\s\S]*?```/g;
  const codeBlocks: string[] = [];
  result = result.replace(codeBlockRegex, (match) => {
    codeBlocks.push(match);
    return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
  });
  
  // Protect list markers (numbers and bullets)
  const listMarkerRegex = /^(\s*)(\d+\.|\-|\*)\s+/gm;
  const listMarkers: string[] = [];
  result = result.replace(listMarkerRegex, (match) => {
    listMarkers.push(match);
    return `__LIST_MARKER_${listMarkers.length - 1}__`;
  });
  
  // Normalize multiple spaces to single space
  result = result.replace(/[ \t]{2,}/g, ' ');
  
  // Restore list markers
  listMarkers.forEach((marker, index) => {
    result = result.replace(`__LIST_MARKER_${index}__`, marker);
  });
  
  // Restore code blocks
  codeBlocks.forEach((block, index) => {
    result = result.replace(`__CODE_BLOCK_${index}__`, block);
  });
  
  return result.trim();
}

// Component to render message content with markdown support
function MessageContent({ content, textColor = '#ececf1' }: { content: string; textColor?: string }) {
  // Normalize content to remove extra spaces and line breaks - memoize to avoid re-computation
  const normalizedContent = useMemo(() => normalizeMarkdown(content), [content]);
  
  return (
    <ReactMarkdown
      components={{
        // Style code blocks
        code: ({ className, children, ...props }: any) => {
          const isInline = !className;
          if (isInline) {
            return (
              <code
                style={{
                  background: '#2d2d2d',
                  padding: '0.15em 0.3em',
                  borderRadius: '3px',
                  fontSize: '0.9em',
                  fontFamily: "'Courier New', monospace",
                  color: '#f8f8f2',
                }}
                {...props}
              >
                {children}
              </code>
            );
          }
          return (
            <pre
              style={{
                background: '#1e1e1e',
                padding: '0.75rem',
                borderRadius: '4px',
                overflowX: 'auto',
                margin: '0.25rem 0',
                border: '1px solid #444',
              }}
            >
              <code
                style={{
                  fontFamily: "'Courier New', monospace",
                  fontSize: '0.875rem',
                  color: '#d4d4d4',
                  whiteSpace: 'pre',
                }}
                {...props}
              >
                {children}
              </code>
            </pre>
          );
        },
        // Style paragraphs - more compact
        p: ({ children }) => (
          <p style={{ margin: '0.25rem 0', lineHeight: '1.5', color: textColor }}>{children}</p>
        ),
        // Style headings - more compact
        h1: ({ children }) => (
          <h1 style={{ fontSize: '1.3rem', fontWeight: 'bold', margin: '0.5rem 0 0.25rem 0', color: textColor }}>{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 style={{ fontSize: '1.15rem', fontWeight: 'bold', margin: '0.4rem 0 0.2rem 0', color: textColor }}>{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 style={{ fontSize: '1.05rem', fontWeight: 'bold', margin: '0.3rem 0 0.15rem 0', color: textColor }}>{children}</h3>
        ),
        // Style lists - more compact
        ul: ({ children }) => (
          <ul style={{ margin: '0.25rem 0', paddingLeft: '1.25rem' }}>{children}</ul>
        ),
        ol: ({ children }) => (
          <ol style={{ margin: '0.25rem 0', paddingLeft: '1.25rem' }}>{children}</ol>
        ),
        li: ({ children }) => (
          <li style={{ margin: '0.15rem 0', color: textColor }}>{children}</li>
        ),
        // Style blockquotes - more compact
        blockquote: ({ children }) => (
          <blockquote
            style={{
              borderLeft: '3px solid #444',
              paddingLeft: '0.75rem',
              margin: '0.25rem 0',
              color: '#aaa',
              fontStyle: 'italic',
            }}
          >
            {children}
          </blockquote>
        ),
        // Style links
        a: ({ children, href }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#19c37d', textDecoration: 'underline' }}
          >
            {children}
          </a>
        ),
        // Style strong/bold
        strong: ({ children }) => (
          <strong style={{ fontWeight: 'bold', color: textColor }}>{children}</strong>
        ),
        // Style emphasis/italic
        em: ({ children }) => (
          <em style={{ fontStyle: 'italic', color: textColor }}>{children}</em>
        ),
      }}
    >
      {normalizedContent}
    </ReactMarkdown>
  );
}
import {
  createSession,
  deleteSession,
  getSession,
  listSessions,
  sendMessage,
} from '../api/client';
import { LLMProviderSelector } from '../components/LLMProviderSelector';
import { ObservabilityPanel } from '../components/ObservabilityPanel';
import { useEventStream, type Event } from '../hooks/useEventStream';
import type {
  MessageResponse,
  SessionResponse,
  SessionSummary,
} from '../types/api';

export function ChatPage() {
  const { sessionId } = useParams<{ sessionId?: string }>();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  // Initialize with empty chat if no sessionId (temporary session until first message)
  const [currentSession, setCurrentSession] = useState<SessionResponse | null>(() => {
    // If no sessionId in URL, create temporary empty session
    return null; // Will be set in useEffect based on sessionId
  });
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  // Removed approving state - no longer needed with direct tool calling
  const [deleting, setDeleting] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [streamingContent, setStreamingContent] = useState<string>(''); // Content being streamed in real-time
  const [intermediateChunks, setIntermediateChunks] = useState<Array<{type: string; content: string; timestamp: number}>>([]); // Intermediate chunks (reasoning, tool calls, etc)
  const [waitingForResponse, setWaitingForResponse] = useState(false); // Track if we're waiting for first chunk
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(false); // Start with auto-scroll disabled
  const loadingSessionRef = useRef<string | null>(null); // Track which session is currently loading
  const loadSessionTimeoutRef = useRef<number | null>(null); // Debounce timer for loadSession
  const failedSessionsRef = useRef<Set<string>>(new Set()); // Track sessions that failed (404)
  const currentSessionRef = useRef<SessionResponse | null>(null); // Ref to track current session for closures
  const pendingStreamingClearRef = useRef<number | null>(null); // Track pending streamingContent clear
  
  // Load sessions list - memoized to avoid unnecessary re-renders
  const loadSessions = useCallback(async () => {
    try {
      const response = await listSessions(50, 0);
      // Filter out sessions that already failed (404) to prevent unnecessary load attempts
      const validSessions = response.sessions.filter(
        (s) => !failedSessionsRef.current.has(s.session_id)
      );
      setSessions((prevSessions) => {
        // Only update if sessions actually changed to avoid re-renders
        if (prevSessions.length === validSessions.length &&
            prevSessions.every((s, idx) => s.session_id === validSessions[idx]?.session_id)) {
          return prevSessions;
        }
        return validSessions;
      });
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }, []);

  // Load current session - simplified: just fetch from backend if not already loaded
  const loadSession = useCallback(async (id: string, skipLoadingState = false, scrollToBottom = false) => {
    // Don't load if already viewing this session (unless forced refresh)
    if (!scrollToBottom && currentSessionRef.current?.session_id === id) {
      console.log('Already viewing this session, skipping load:', id);
      return;
    }
    
    // Don't load if this session already failed (404)
    if (failedSessionsRef.current.has(id)) {
      console.log('Skipping load for failed session:', id);
      return;
    }
    
    // Don't load if already loading this session
    if (loadingSessionRef.current === id) {
      console.log('Already loading session:', id);
      return;
    }
    
    // Clear any pending debounce timer
    if (loadSessionTimeoutRef.current !== null) {
      clearTimeout(loadSessionTimeoutRef.current);
      loadSessionTimeoutRef.current = null;
    }
    
    // Preserve sidebar scroll position
    const sidebarScrollTop = sidebarRef.current?.scrollTop ?? 0;
    
    // Mark as loading
    loadingSessionRef.current = id;
    
    if (!skipLoadingState) {
      setLoading(true);
    }
    setError(null);
    
    try {
      // Fetch session from backend
      const session = await getSession(id);
      
      // Session loaded successfully - remove from failed set if it was there
      failedSessionsRef.current.delete(id);
      
      // Always update session - the comparison was too strict and prevented updates
      // The streaming content will be cleared if it matches the final message
      setCurrentSession(session);
      // Update ref for use in closures
      currentSessionRef.current = session;
      
      // Restore sidebar scroll position
      requestAnimationFrame(() => {
        if (sidebarRef.current) {
          sidebarRef.current.scrollTop = sidebarScrollTop;
        }
        
        // If scrollToBottom is true, always scroll to bottom instantly (no animation)
        if (scrollToBottom && messagesContainerRef.current) {
          // Use double requestAnimationFrame to ensure DOM is updated
          requestAnimationFrame(() => {
            if (messagesContainerRef.current) {
              messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
            }
          });
        }
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load session';
      
      // Check if it's a 404 error
      if (errorMessage.includes('404') || errorMessage.includes('Not Found')) {
        console.warn('Session not found (404):', id);
        // Mark session as failed to prevent future attempts
        failedSessionsRef.current.add(id);
        // Clear current session if it matches
        setCurrentSession((prevSession) => {
          if (prevSession && prevSession.session_id === id) {
            currentSessionRef.current = null; // Update ref
            return null;
          }
          return prevSession;
        });
        setError(`Session not found`);
      } else {
        setError(errorMessage);
      }
    } finally {
      loadingSessionRef.current = null;
      if (!skipLoadingState) {
        setLoading(false);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // No dependencies - loadSession is stable and uses refs for all dynamic values
  
  // Handle WebSocket events - memoized to avoid unnecessary re-renders
  const handleEvent = useCallback((event: Event) => {
    // STRICT: Only process events for current session
    // Ignore events without session_id or for different sessions
    const eventSessionId = event.properties?.session_id;
    if (!eventSessionId) {
      // Events without session_id are system events (heartbeat, etc.) - ignore
      return;
    }
    
    // Don't process events if current session is null
    if (!currentSession) {
      return;
    }
    
    // STRICT: Only process events for the exact current session
    if (eventSessionId !== currentSession.session_id) {
      // Silently ignore events from other sessions (including deleted ones)
      // Log only if it's a deleted session to help debug
      if (failedSessionsRef.current.has(eventSessionId)) {
        console.debug('Ignoring event from deleted session:', eventSessionId, event.type);
      }
      return;
    }
    
    // Don't process events for sessions that already failed (404)
    if (failedSessionsRef.current.has(eventSessionId)) {
      console.debug('Ignoring event for failed session:', eventSessionId, event.type);
      return;
    }
    
    // Only log events that we actually process
    console.log('Processing event:', event.type, 'for session:', eventSessionId);
    
    // NO MORE loadSession during runtime - only update state from SSE events
    
    switch (event.type) {
      case 'session.message.added':
        // User message was added - update state locally
        // For user messages, we already have them from sendMessage
        // For assistant messages, they come via llm.stream.token
        // No need to reload from backend
        break;
      
      case 'session.updated':
        // Session metadata updated - don't update title (keep original title)
        // No need to reload full session - messages come via SSE
        // Title should remain as originally set, not change to first message
        break;
      
      case 'execution.step.completed':
        // Execution step completed - shown via intermediate chunks
        // No need to reload - all updates come via SSE
        break;
      
      case 'execution.completed':
        // Execution completed - all messages already streamed via llm.stream.token
        // No need to reload - state is updated from events
        break;
      
      case 'execution.failed':
        // Execution failed - show error in UI without reloading
        // Error information should be in event.properties
        if (event.properties?.error) {
          setError(event.properties.error);
        }
        break;
      
      case 'planner.thinking':
        // LLM is thinking - show raw response in console or UI
        console.log('LLM thinking:', event.properties.raw_response);
        // Optionally show in UI (you can add a "thinking" indicator)
        break;
      
      case 'llm.reasoning':
        // LLM is reasoning/thinking - store as intermediate chunk
        const reasoningContent = event.properties.content || event.properties.message;
        console.log('🤔 LLM reasoning:', reasoningContent);
        if (reasoningContent) {
          setIntermediateChunks((prev) => [
            ...prev,
            { type: 'reasoning', content: String(reasoningContent), timestamp: Date.now() }
          ]);
        }
        break;
      
      case 'llm.stream.start':
        // LLM generation started - initialize streaming content
        console.log('💬 LLM generation started');
        setStreamingContent(''); // Reset streaming content for new message
        setIntermediateChunks([]); // Reset intermediate chunks for new message
        setWaitingForResponse(true); // Show loading indicator until first chunk arrives
        shouldAutoScrollRef.current = true; // Enable auto-scroll during streaming
        
        // Don't create placeholder message yet - wait for first token
        break;
      
      case 'llm.stream.token':
        // Real-time token streaming - update UI in real-time
        // Process content_blocks pattern (preferred) or legacy token format
        let newToken = '';
        
        if (event.properties.content_blocks) {
          // Process content_blocks pattern (reasoning, tool_call_chunk, text)
          const blocks = event.properties.content_blocks;
          for (const block of blocks) {
            if (block.type === 'reasoning' && block.reasoning) {
              console.log('🤔 Reasoning:', block.reasoning);
              // Store reasoning as intermediate chunk
              setIntermediateChunks((prev) => [
                ...prev,
                { type: 'reasoning', content: block.reasoning, timestamp: Date.now() }
              ]);
            } else if (block.type === 'tool_call_chunk') {
              console.log('🔧 Tool call chunk:', block.tool_call);
              // Store tool call as intermediate chunk
              const toolInfo = block.tool_call?.name || 'unknown';
              const toolArgs = JSON.stringify(block.tool_call?.args || {});
              setIntermediateChunks((prev) => [
                ...prev,
                { type: 'tool_call', content: `${toolInfo}(${toolArgs})`, timestamp: Date.now() }
              ]);
            } else if (block.type === 'text' && block.text) {
              newToken += block.text; // Accumulate text tokens
            }
          }
        } else if (event.properties.token) {
          // Legacy token format (backward compatibility) - only use if content_blocks not available
          newToken = event.properties.token;
        }
        
        // Update streaming content in real-time (only if we have new tokens)
        if (newToken && currentSession) {
          setStreamingContent((prev) => {
            const updated = prev + newToken;
            
            // On first token, create assistant message and hide loading indicator
            const messageId = event.properties.message_id;
            if (messageId) {
              setCurrentSession((prevSession) => {
                if (!prevSession) return prevSession;
                const messageIndex = prevSession.messages.findIndex(m => m.message_id === messageId);
                
                if (messageIndex >= 0) {
                  // Update existing message
                  const updatedMessages = [...prevSession.messages];
                  updatedMessages[messageIndex] = {
                    ...updatedMessages[messageIndex],
                    content: updated,
                  };
                  return {
                    ...prevSession,
                    messages: updatedMessages,
                  };
                } else {
                  // First token - create assistant message
                  // Hide loading indicator now that we have content
                  setWaitingForResponse(false);
                  
                  const assistantMessage: MessageResponse = {
                    message_id: messageId,
                    role: 'assistant',
                    content: updated,
                    created_at: Date.now(),
                  };
                  
                  return {
                    ...prevSession,
                    messages: [...prevSession.messages, assistantMessage],
                  };
                }
              });
            }
            
            // Auto-scroll to bottom during streaming - use requestAnimationFrame to avoid flickering
            requestAnimationFrame(() => {
              if (shouldAutoScrollRef.current && messagesEndRef.current) {
                messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
              }
            });
            return updated;
          });
        }
        break;
      
      case 'llm.stream.end':
        // LLM generation completed - reload full session from backend
        console.log('💬 LLM generation completed - will reload session from backend after delay');
        if (event.properties.reasoning) {
          console.log('🤔 Reasoning:', event.properties.reasoning);
          // Store final reasoning if available
          setIntermediateChunks((prev) => [
            ...prev,
            { type: 'reasoning', content: event.properties.reasoning, timestamp: Date.now() }
          ]);
        }
        
        // Clear streaming content and loading indicator
        setStreamingContent('');
        setWaitingForResponse(false);
        if (pendingStreamingClearRef.current !== null) {
          clearTimeout(pendingStreamingClearRef.current);
          pendingStreamingClearRef.current = null;
        }
        
        // Reload full session from backend to ensure we have the complete conversation
        // Add delay to give backend time to save everything to disk
        const eventSessionId = event.properties?.session_id;
        if (eventSessionId && currentSession && currentSession.session_id === eventSessionId) {
          // Wait a bit for backend to finish saving to disk before reloading
          // This ensures we get the complete conversation with all messages saved
          setTimeout(() => {
            // Reload session from backend with scrollToBottom=true to show the end
            loadSession(eventSessionId, true, true).catch((err) => {
              console.error('Failed to reload session after LLM completion:', err);
            });
          }, 500); // 500ms delay to ensure backend has saved everything
        }
        break;
      
      case 'tool.decision':
        // Model decided to use or not use tools
        console.log(`Tool decision: ${event.properties.decision}`, event.properties.reasoning);
        if (event.properties.decision === 'use_tool') {
          console.log(`  → Will call ${event.properties.tool_calls_count} tool(s)`);
        } else {
          console.log(`  → Responding directly without tools`);
        }
        break;
      
      case 'tool.called':
      case 'tool.stream.start':
        // Tool is being called - store as intermediate chunk
        const toolName = event.properties.tool || 'unknown';
        const toolArgs = event.properties.arguments || event.properties.input || {};
        console.log(`🔧 Calling tool: ${toolName}`, toolArgs);
        setIntermediateChunks((prev) => [
          ...prev,
          { 
            type: 'tool_call', 
            content: `${toolName}(${JSON.stringify(toolArgs)})`, 
            timestamp: Date.now() 
          }
        ]);
        break;
      
      case 'tool.stream.end':
        // Tool execution completed
        console.log(`✅ Tool completed: ${event.properties.tool}`);
        console.log(`   Output: ${event.properties.output?.substring(0, 200)}...`);
        break;
      
      case 'tool.stream.error':
        // Tool execution error
        console.log(`❌ Tool error: ${event.properties.tool} - ${event.properties.error}`);
        break;
      
      case 'chain.stream.start':
        // Chain execution started
        console.log(`🔄 Chain started: ${event.properties.chain}`);
        break;
      
      case 'chain.stream.end':
        // Chain execution completed
        console.log(`🔄 Chain completed: ${event.properties.chain}`);
        break;
      
      case 'tool.result':
        // Tool result received - show progress
        console.log(`Tool result: ${event.properties.tool}.${event.properties.operation} - ${event.properties.success ? 'success' : 'failed'}`);
        // Don't reload during streaming - tool results are intermediate steps
        // The final result will come via execution.completed or session.updated
        break;
      
      default:
        // Ignore other events
        break;
    }
  }, [currentSession, streamingContent]);
  
  // Connect to WebSocket event stream (only when session is active and has valid sessionId)
  // Pass session_id to filter events - only receive events for current session
  // Don't connect if sessionId is empty (temporary session before first message)
  const hasValidSession = !!(currentSession?.session_id && currentSession.session_id !== '');
  useEventStream(handleEvent, hasValidSession, currentSession?.session_id);

  // Create new session (clear current chat and start fresh)
  const handleNewSession = () => {
    // Clear streaming state
    setStreamingContent('');
    setIntermediateChunks([]);
    setError(null);
    
    // Create temporary empty session (will be created when user sends first message)
    const tempSession: SessionResponse = {
      session_id: '', // Empty - will be set when session is created
      title: 'New Chat',
      messages: [],
      created_at: Date.now(),
      updated_at: Date.now(),
    };
    setCurrentSession(tempSession);
    currentSessionRef.current = tempSession;
    
    // Navigate to root (no sessionId) to show empty chat
    window.history.pushState({}, '', '/chat');
    
    // Don't reload sessions list - it's not necessary for just clearing the chat
    // Sessions list is already loaded on mount and will be updated when new session is created
  };

  // Send message
  const handleSend = async () => {
    if (!input.trim() || sending) return;
    
    // If no currentSession, create temporary one for UI
    if (!currentSession) {
      const tempSession: SessionResponse = {
        session_id: '', // Will be set after session creation
        title: 'New Chat',
        messages: [],
        created_at: Date.now(),
        updated_at: Date.now(),
      };
      setCurrentSession(tempSession);
      currentSessionRef.current = tempSession;
    }

    const messageContent = input.trim();
    setInput('');
    setStreamingContent(''); // Clear any previous streaming content
    setSending(true);
    setError(null);

    // Step 1: Create session if it doesn't exist (first message)
    let actualSessionId = currentSession?.session_id || '';
    if (!actualSessionId || actualSessionId === '') {
      try {
        const newSession = await createSession();
        actualSessionId = newSession.session_id;
        
        // Update current session with real sessionId
        const sessionWithId: SessionResponse = {
          ...currentSession!,
          session_id: actualSessionId,
          title: newSession.title,
          messages: currentSession?.messages || [],
        };
        setCurrentSession(sessionWithId);
        currentSessionRef.current = sessionWithId;
        
        // Update URL to include sessionId
        window.history.pushState({}, '', `/chat/${actualSessionId}`);
        
        // Reload sessions list to include new session
        await loadSessions();
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to create session');
        setSending(false);
        return;
      }
    }

    // Step 2: Add user message optimistically
    const userMessage: MessageResponse = {
      message_id: `temp-${Date.now()}`,
      role: 'user',
      content: messageContent,
      created_at: Date.now(),
    };

    // When user sends a message, enable auto-scroll and scroll to bottom
    shouldAutoScrollRef.current = true;
    
    // Update session with user message
    const updatedSession = {
      ...currentSession!,
      session_id: actualSessionId,
      messages: [...currentSession!.messages, userMessage],
    };
    setCurrentSession(updatedSession);
    currentSessionRef.current = updatedSession; // Update ref
    
    // Scroll to bottom to show user message immediately
    requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTo({
          top: messagesContainerRef.current.scrollHeight,
          behavior: 'smooth'
        });
      }
    });

    try {
      // Step 3: Send message (non-blocking, returns immediately)
      await sendMessage(actualSessionId, messageContent);
      // When agent responds, disable auto-scroll so user can manually scroll
      shouldAutoScrollRef.current = false;
      // Don't load session here - events will come via SSE
      // Don't refresh session list - only update on mount or when explicitly needed
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      // Remove optimistic message on error
      const revertedSession = {
        ...updatedSession,
        messages: updatedSession.messages.slice(0, -1),
      };
      setCurrentSession(revertedSession);
      currentSessionRef.current = revertedSession; // Update ref
    } finally {
      setSending(false);
    }
  };

  // Memoize messages length to avoid unnecessary re-renders and scrollbar flickering
  const messagesLength = useMemo(() => currentSession?.messages.length ?? 0, [currentSession?.messages.length]);
  
  // Scroll to bottom smoothly when messages change - ONLY if auto-scroll is enabled
  // This happens when user sends a message, but NOT when agent responds
  // Use messagesLength instead of messages array to avoid flickering
  useEffect(() => {
    if (!messagesContainerRef.current) return;
    
    // Only scroll if auto-scroll is explicitly enabled (user just sent a message)
    if (shouldAutoScrollRef.current) {
      // Use requestAnimationFrame to avoid layout thrashing
      const rafId = requestAnimationFrame(() => {
        if (messagesContainerRef.current && shouldAutoScrollRef.current) {
          messagesContainerRef.current.scrollTo({
            top: messagesContainerRef.current.scrollHeight,
            behavior: 'smooth'
          });
        }
      });
      
      return () => cancelAnimationFrame(rafId);
    }
    // When agent responds, auto-scroll is disabled, so scroll position stays where user left it
  }, [messagesLength]);

  // Track scroll position to determine if we should auto-scroll
  const handleMessagesScroll = () => {
    if (!messagesContainerRef.current) return;
    
    const container = messagesContainerRef.current;
    const isNearBottom = 
      container.scrollHeight - container.scrollTop - container.clientHeight < 150;
    
    // Update shouldAutoScroll based on scroll position
    shouldAutoScrollRef.current = isNearBottom;
  };

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Sync currentSessionRef with currentSession state to prevent stale closures
  // BUT: Only update ref, don't trigger any side effects that could cause navigation
  useEffect(() => {
    currentSessionRef.current = currentSession;
  }, [currentSession]);

  // Load session when sessionId changes - ONLY when sessionId actually changes
  // DO NOT include loadSession in dependencies to prevent unnecessary re-runs
  useEffect(() => {
    // Don't try to load sessions that already failed (404)
    if (sessionId && failedSessionsRef.current.has(sessionId)) {
      console.log('Skipping load for failed session in useEffect:', sessionId);
      setCurrentSession(null);
      currentSessionRef.current = null; // Update ref
      return;
    }
    
    // Don't load if already viewing this session (avoid unnecessary reloads)
    // Use ref only to avoid dependency on currentSession state (prevents unnecessary triggers)
    const currentSessionId = currentSessionRef.current?.session_id;
    if (sessionId && currentSessionId === sessionId) {
      console.log('Already viewing this session in useEffect, skipping load:', sessionId);
      return;
    }
    
    // Clear failed sessions when navigating to a new session (only if it's different)
    if (sessionId && currentSessionId && currentSessionId !== sessionId) {
      failedSessionsRef.current.clear();
    }
    
    if (sessionId) {
      loadSession(sessionId, false, true); // Use loading state for initial load, scroll to bottom
    } else {
      // If no sessionId, create temporary empty session for initial chat
      const tempSession: SessionResponse = {
        session_id: '', // Empty - will be set when session is created
        title: 'New Chat',
        messages: [],
        created_at: Date.now(),
        updated_at: Date.now(),
      };
      setCurrentSession(tempSession);
      currentSessionRef.current = tempSession;
    }
    
    // Cleanup: clear timeout on unmount or session change
    return () => {
      if (loadSessionTimeoutRef.current !== null) {
        clearTimeout(loadSessionTimeoutRef.current);
        loadSessionTimeoutRef.current = null;
      }
      loadingSessionRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]); // Only depend on sessionId - loadSession is stable and doesn't need to be in deps

  // Handle Enter key
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Removed handleApprove and handleReject - no longer needed with direct tool calling

  // Handle delete session - show confirmation icons
  const handleDeleteClick = async (sessionIdToDelete: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent navigation when clicking delete
    setConfirmingDelete(sessionIdToDelete);
  };

  // Handle confirm delete
  const handleConfirmDelete = async (sessionIdToDelete: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmingDelete(null);
    setDeleting(sessionIdToDelete);
    setError(null);

    try {
      await deleteSession(sessionIdToDelete);
      // Reload sessions list
      await loadSessions();
      
      // If deleted session was the current one, navigate to home
      if (currentSession && currentSession.session_id === sessionIdToDelete) {
        window.history.pushState({}, '', '/chat');
        setCurrentSession(null);
        currentSessionRef.current = null; // Update ref
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete session');
    } finally {
      setDeleting(null);
    }
  };

  // Handle cancel delete
  const handleCancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setConfirmingDelete(null);
  };

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        overflow: 'hidden',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      {/* Sidebar */}
      <div
        ref={sidebarRef}
        style={{
          width: '260px',
          background: '#202123',
          color: '#fff',
          display: 'flex',
          flexDirection: 'column',
          borderRight: '1px solid #444',
        }}
      >
        <div style={{ padding: '1rem' }}>
          <button
            onClick={handleNewSession}
            disabled={sending}
            style={{
              width: '100%',
              padding: '0.75rem',
              background: sending ? '#444' : '#19c37d',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: sending ? 'not-allowed' : 'pointer',
              fontSize: '0.875rem',
              fontWeight: '500',
            }}
          >
            + New Chat
          </button>
        </div>

        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '0.5rem',
          }}
        >
          {sessions.map((session) => (
            <div
              key={session.session_id}
              onClick={() => {
                // Don't navigate to sessions that already failed (404)
                if (failedSessionsRef.current.has(session.session_id)) {
                  console.log('Skipping navigation to failed session:', session.session_id);
                  return;
                }
                
                // Don't do anything if already viewing this session
                if (currentSession?.session_id === session.session_id) {
                  console.log('Already viewing this session, skipping reload');
                  return;
                }
                
                window.history.pushState({}, '', `/chat/${session.session_id}`);
                loadSession(session.session_id, false, true); // Scroll to bottom when changing chats
              }}
              style={{
                padding: '0.75rem',
                marginBottom: '0.25rem',
                borderRadius: '6px',
                cursor: 'pointer',
                background:
                  currentSession?.session_id === session.session_id ? '#343541' : 'transparent',
                color: '#fff',
                fontSize: '0.875rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '0.5rem',
              }}
              onMouseEnter={(e) => {
                // Make text bold on hover
                const span = e.currentTarget.querySelector('span');
                if (span) {
                  span.style.fontWeight = '600';
                }
              }}
              onMouseLeave={(e) => {
                // Remove bold on leave
                const span = e.currentTarget.querySelector('span');
                if (span) {
                  span.style.fontWeight = 'normal';
                }
              }}
            >
              <span
                style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                  fontWeight: 'normal',
                }}
              >
                {session.session_id}
              </span>
              {confirmingDelete === session.session_id ? (
                <div style={{ display: 'flex', gap: '0.25rem', alignItems: 'center' }}>
                  <button
                    onClick={handleCancelDelete}
                    disabled={deleting === session.session_id}
                    style={{
                      padding: '0',
                      background: 'transparent',
                      color: '#dc3545',
                      border: 'none',
                      cursor: deleting === session.session_id ? 'not-allowed' : 'pointer',
                      fontSize: '1rem',
                      flexShrink: 0,
                      width: '20px',
                      height: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: deleting === session.session_id ? 0.5 : 1,
                    }}
                    title="Cancel delete"
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = '#e74c3c';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = '#dc3545';
                    }}
                  >
                    ✕
                  </button>
                  <button
                    onClick={(e) => handleConfirmDelete(session.session_id, e)}
                    disabled={deleting === session.session_id}
                    style={{
                      padding: '0',
                      background: 'transparent',
                      color: '#19c37d',
                      border: 'none',
                      cursor: deleting === session.session_id ? 'not-allowed' : 'pointer',
                      fontSize: '1rem',
                      flexShrink: 0,
                      width: '20px',
                      height: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: deleting === session.session_id ? 0.5 : 1,
                    }}
                    title="Confirm delete"
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = '#1dd085';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = '#19c37d';
                    }}
                  >
                    ✓
                  </button>
                </div>
              ) : (
                <button
                  onClick={(e) => handleDeleteClick(session.session_id, e)}
                  disabled={deleting === session.session_id}
                  style={{
                    padding: '0',
                    background: 'transparent',
                    color: '#888',
                    border: 'none',
                    cursor: deleting === session.session_id ? 'not-allowed' : 'pointer',
                    fontSize: '1rem',
                    flexShrink: 0,
                    width: '20px',
                    height: '20px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    opacity: deleting === session.session_id ? 0.5 : 1,
                  }}
                  title="Delete session"
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = '#fff';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = '#888';
                  }}
                >
                  {deleting === session.session_id ? '...' : '✕'}
                </button>
              )}
            </div>
          ))}
        </div>

        {/* LLM Provider Selector - at the bottom */}
        <div
          style={{
            borderTop: '1px solid #444',
            padding: '1rem',
            background: '#202123',
          }}
        >
          <LLMProviderSelector />
        </div>
      </div>

      {/* Main chat area */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'row',
          background: '#343541',
          color: '#fff',
        }}
      >
        {/* Chat content */}
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            minWidth: 0,
          }}
        >
        {error && (
          <div
            style={{
              padding: '1rem',
              background: '#dc3545',
              color: '#fff',
              textAlign: 'center',
            }}
          >
            {error}
          </div>
        )}

        {loading ? (
          <div
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            Loading...
          </div>
        ) : currentSession ? (
          <>
            {/* Messages */}
            <div
              ref={messagesContainerRef}
              onScroll={handleMessagesScroll}
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '1rem 1.5rem',
                maxWidth: '768px',
                margin: '0 auto',
                width: '100%',
              }}
            >
              {currentSession.messages.map((message) => (
                <div
                  key={message.message_id}
                  style={{
                    marginBottom: '1rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.25rem',
                  }}
                >
                  <div
                    style={{
                      fontWeight: '600',
                      fontSize: '0.875rem',
                      color: message.role === 'user' ? '#19c37d' : '#8b9dc3',
                    }}
                  >
                    {message.role === 'user' ? 'You' : 'Assistant'}
                  </div>
                  <div
                    style={{
                      whiteSpace: 'pre-wrap',
                      lineHeight: '1.5',
                    }}
                  >
                    <MessageContent 
                      content={message.content} 
                      textColor={message.role === 'user' ? '#d4d4d4' : '#ececf1'}
                    />
                  </div>
                  {/* Removed plan_result display - no longer used with direct tool calling */}
                </div>
              ))}
              {/* Show loading indicator while waiting for first chunk */}
              {waitingForResponse && !streamingContent && (
                <div
                  style={{
                    marginBottom: '1rem',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    width: '100%',
                  }}
                >
                  <div
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: '#19c37d',
                      animation: 'pulse 1.5s ease-in-out infinite',
                    }}
                  />
                </div>
              )}
              {/* Show streaming content in real-time while streaming */}
              {/* The message is also updated in state by tokens, but we show streamingContent for real-time updates */}
              {/* streamingContent will be cleared after llm.stream.end, but by then the message is in state */}
              {streamingContent && (
                <div
                  style={{
                    marginBottom: '1rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.25rem',
                  }}
                >
                  <div
                    style={{
                      fontWeight: '600',
                      fontSize: '0.875rem',
                      color: '#8b9dc3',
                    }}
                  >
                    Assistant
                  </div>
                  <div
                    style={{
                      whiteSpace: 'pre-wrap',
                      lineHeight: '1.5',
                    }}
                  >
                    <MessageContent content={streamingContent} textColor="#ececf1" />
                  </div>
                  {/* Show intermediate chunks (reasoning, tool calls) in smaller, gray text */}
                  {intermediateChunks.length > 0 && (
                    <div
                      style={{
                        marginTop: '0.5rem',
                        paddingTop: '0.5rem',
                        borderTop: '1px solid #444',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '0.25rem',
                      }}
                    >
                      {intermediateChunks.map((chunk, index) => (
                        <div
                          key={index}
                          style={{
                            fontSize: '0.75rem',
                            color: '#888',
                            fontFamily: 'monospace',
                            whiteSpace: 'pre-wrap',
                            lineHeight: '1.4',
                          }}
                        >
                          {chunk.type === 'reasoning' && (
                            <span>
                              <span style={{ color: '#666' }}>🤔 Reasoning: </span>
                              {chunk.content}
                            </span>
                          )}
                          {chunk.type === 'tool_call' && (
                            <span>
                              <span style={{ color: '#666' }}>🔧 Tool: </span>
                              {chunk.content}
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input area */}
            <div
              style={{
                padding: '1rem',
                borderTop: '1px solid #444',
                maxWidth: '768px',
                margin: '0 auto',
                width: '100%',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  gap: '0.5rem',
                  background: '#40414f',
                  borderRadius: '12px',
                  padding: '0.75rem',
                  border: '1px solid #565869',
                }}
              >
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message..."
                  disabled={sending}
                  style={{
                    flex: 1,
                    background: 'transparent',
                    border: 'none',
                    color: '#fff',
                    fontSize: '1rem',
                    resize: 'none',
                    outline: 'none',
                    minHeight: '24px',
                    maxHeight: '200px',
                    fontFamily: 'inherit',
                  }}
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || sending}
                  style={{
                    padding: '0.5rem 1rem',
                    background: input.trim() && !sending ? '#19c37d' : '#444',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: input.trim() && !sending ? 'pointer' : 'not-allowed',
                    fontSize: '0.875rem',
                  }}
                >
                  {sending ? 'Sending...' : 'Send'}
                </button>
              </div>
            </div>
          </>
        ) : null}
        </div>

        {/* Observability panel */}
        <ObservabilityPanel />
      </div>
    </div>
  );
}
