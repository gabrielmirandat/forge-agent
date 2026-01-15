/**Chat page - main chat interface with sidebar.

Layout similar to ChatGPT:
- Left sidebar with session history
- Main area with chat messages
- Input at bottom
*/

import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';

// Component to render message content with code blocks
function MessageContent({ content }: { content: string }) {
  const parts: (string | { type: 'code'; content: string })[] = [];
  let lastIndex = 0;
  const codeBlockRegex = /```([\s\S]*?)```/g;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      const text = content.slice(lastIndex, match.index);
      if (text) {
        parts.push(text);
      }
    }
    // Add code block
    parts.push({ type: 'code', content: match[1].trim() });
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }

  // If no code blocks found, return original content
  if (parts.length === 0) {
    return <>{content}</>;
  }

  return (
    <>
      {parts.map((part, index) => {
        if (typeof part === 'string') {
          return <span key={index}>{part}</span>;
        } else {
          return (
            <pre
              key={index}
              style={{
                background: '#1e1e1e',
                padding: '1rem',
                borderRadius: '6px',
                overflowX: 'auto',
                margin: '0.5rem 0',
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
              >
                {part.content}
              </code>
            </pre>
          );
        }
      })}
    </>
  );
}
import {
  approveOperations,
  createSession,
  deleteSession,
  getSession,
  listSessions,
  rejectOperations,
  sendMessage,
} from '../api/client';
import { ObservabilityPanel } from '../components/ObservabilityPanel';
import type {
  MessageResponse,
  SessionResponse,
  SessionSummary,
} from '../types/api';

export function ChatPage() {
  const { sessionId } = useParams<{ sessionId?: string }>();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [currentSession, setCurrentSession] = useState<SessionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [approving, setApproving] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(false); // Start with auto-scroll disabled

  // Load sessions list
  const loadSessions = async () => {
    try {
      const response = await listSessions(50, 0);
      setSessions(response.sessions);
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  };

  // Load current session
  const loadSession = async (id: string) => {
    // Preserve sidebar scroll position
    const sidebarScrollTop = sidebarRef.current?.scrollTop ?? 0;
    
    setLoading(true);
    setError(null);
    try {
      const session = await getSession(id);
      setCurrentSession(session);
      
      // Restore sidebar scroll position after state update
      setTimeout(() => {
        if (sidebarRef.current) {
          sidebarRef.current.scrollTop = sidebarScrollTop;
        }
      }, 0);
      
      // Don't auto-scroll when loading a session - preserve user's scroll position
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session');
    } finally {
      setLoading(false);
    }
  };

  // Create new session
  const handleNewSession = async () => {
    setSending(true);
    setError(null);
    try {
      const response = await createSession();
      await loadSessions();
      // Navigate to new session (will trigger useEffect)
      window.history.pushState({}, '', `/chat/${response.session_id}`);
      await loadSession(response.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session');
    } finally {
      setSending(false);
    }
  };

  // Send message
  const handleSend = async () => {
    if (!input.trim() || !currentSession || sending) return;

    const messageContent = input.trim();
    setInput('');
    setSending(true);
    setError(null);

    // Optimistically add user message
    const userMessage: MessageResponse = {
      message_id: `temp-${Date.now()}`,
      role: 'user',
      content: messageContent,
      created_at: Date.now() / 1000,
      plan_result: null,
      execution_result: null,
    };

    // When user sends a message, enable auto-scroll and scroll to bottom
    shouldAutoScrollRef.current = true;
    
    // Scroll to bottom immediately after adding user message
    setCurrentSession({
      ...currentSession,
      messages: [...currentSession.messages, userMessage],
    });
    
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
      await sendMessage(currentSession.session_id, messageContent);
      // When agent responds, disable auto-scroll so user can manually scroll
      shouldAutoScrollRef.current = false;
      await loadSession(currentSession.session_id);
      await loadSessions(); // Refresh list to update titles
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      // Remove optimistic message on error
      setCurrentSession({
        ...currentSession,
        messages: currentSession.messages.slice(0, -1),
      });
    } finally {
      setSending(false);
    }
  };

  // Scroll to bottom smoothly when messages change - ONLY if auto-scroll is enabled
  // This happens when user sends a message, but NOT when agent responds
  useEffect(() => {
    if (!messagesContainerRef.current) return;
    
    // Only scroll if auto-scroll is explicitly enabled (user just sent a message)
    if (shouldAutoScrollRef.current) {
      requestAnimationFrame(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTo({
            top: messagesContainerRef.current.scrollHeight,
            behavior: 'smooth'
          });
        }
      });
    }
    // When agent responds, auto-scroll is disabled, so scroll position stays where user left it
  }, [currentSession?.messages]);

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

  // Load session when sessionId changes
  useEffect(() => {
    if (sessionId) {
      loadSession(sessionId);
    } else {
      setCurrentSession(null);
    }
  }, [sessionId]);

  // Handle Enter key
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Handle approve operations
  const handleApprove = async (messageId: string, stepIds: number[]) => {
    if (!currentSession || approving) return;

    setApproving(messageId);
    setError(null);

    try {
      // Preserve scroll position
      const wasAtBottom = shouldAutoScrollRef.current;
      await approveOperations(currentSession.session_id, messageId, stepIds);
      // Reload session to get updated messages
      await loadSession(currentSession.session_id);
      // Restore auto-scroll state
      shouldAutoScrollRef.current = wasAtBottom;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve operations');
    } finally {
      setApproving(null);
    }
  };

  // Handle reject operations
  const handleReject = async (messageId: string, stepIds: number[]) => {
    if (!currentSession || approving) return;

    setApproving(messageId);
    setError(null);

    try {
      // Preserve scroll position
      const wasAtBottom = shouldAutoScrollRef.current;
      await rejectOperations(currentSession.session_id, messageId, stepIds);
      // Reload session to get updated messages
      await loadSession(currentSession.session_id);
      // Restore auto-scroll state
      shouldAutoScrollRef.current = wasAtBottom;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject operations');
    } finally {
      setApproving(null);
    }
  };

  // Handle delete session - show confirmation icons
  const handleDeleteClick = (sessionIdToDelete: string, e: React.MouseEvent) => {
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
                window.history.pushState({}, '', `/chat/${session.session_id}`);
                loadSession(session.session_id);
              }}
              style={{
                padding: '0.75rem',
                marginBottom: '0.25rem',
                borderRadius: '6px',
                cursor: 'pointer',
                background:
                  sessionId === session.session_id ? '#343541' : 'transparent',
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
                {session.title}
              </span>
              {confirmingDelete === session.session_id ? (
                <div style={{ display: 'flex', gap: '0.25rem', alignItems: 'center' }}>
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
        ) : !currentSession ? (
          <div
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              gap: '1rem',
            }}
          >
            <h1 style={{ fontSize: '2rem', margin: 0 }}>Forge Agent</h1>
            <p style={{ color: '#888' }}>Start a new conversation</p>
            <button
              onClick={handleNewSession}
              disabled={sending}
              style={{
                padding: '0.75rem 1.5rem',
                background: sending ? '#444' : '#19c37d',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: sending ? 'not-allowed' : 'pointer',
                fontSize: '1rem',
              }}
            >
              New Chat
            </button>
          </div>
        ) : (
          <>
            {/* Messages */}
            <div
              ref={messagesContainerRef}
              onScroll={handleMessagesScroll}
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '2rem',
                maxWidth: '768px',
                margin: '0 auto',
                width: '100%',
              }}
            >
              {currentSession.messages.map((message) => (
                <div
                  key={message.message_id}
                  style={{
                    marginBottom: '2rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.5rem',
                  }}
                >
                  <div
                    style={{
                      fontWeight: '600',
                      fontSize: '0.875rem',
                      color: message.role === 'user' ? '#19c37d' : '#fff',
                    }}
                  >
                    {message.role === 'user' ? 'You' : 'Assistant'}
                  </div>
                  <div
                    style={{
                      whiteSpace: 'pre-wrap',
                      lineHeight: '1.6',
                      color: '#ececf1',
                    }}
                  >
                    <MessageContent content={message.content} />
                  </div>
                  {message.plan_result && (
                    <div
                      style={{
                        marginTop: '0.5rem',
                        padding: '0.75rem',
                        background: '#40414f',
                        borderRadius: '6px',
                        fontSize: '0.875rem',
                      }}
                    >
                      <div style={{ marginBottom: '0.5rem' }}>
                        <strong>Plan:</strong> {message.plan_result.plan.objective}
                      </div>
                      {message.plan_result.plan.steps.length > 0 && (
                        <div style={{ marginTop: '0.5rem' }}>
                          <strong>Steps:</strong>
                          <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                            {message.plan_result.plan.steps.map((step) => {
                              const requiresApproval = message.pending_approval_steps?.includes(step.step_id);
                              const isRestricted = message.restricted_steps?.includes(step.step_id);
                              return (
                                <div
                                  key={step.step_id}
                                  style={{
                                    padding: '0.5rem',
                                    background: isRestricted ? '#4a1a1a' : requiresApproval ? '#7a1a1a' : '#2a2b2f',
                                    borderRadius: '4px',
                                    border: isRestricted ? '2px solid #ff0000' : requiresApproval ? '1px solid #ff4444' : '1px solid transparent',
                                  }}
                                >
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <div>
                                      <strong>Step {step.step_id}:</strong> {step.tool}.{step.operation}
                                      {isRestricted && (
                                        <span style={{ marginLeft: '0.5rem', color: '#ff6666', fontWeight: 'bold' }}>🚫 RESTRICTED COMMAND</span>
                                      )}
                                      {requiresApproval && !isRestricted && (
                                        <span style={{ marginLeft: '0.5rem', color: '#ff8888' }}>⚠️ Requires Approval</span>
                                      )}
                                    </div>
                                  </div>
                                  <div style={{ marginTop: '0.25rem', fontSize: '0.8rem', color: '#aaa' }}>
                                    {step.rationale}
                                  </div>
                                  {isRestricted && (
                                    <div style={{ marginTop: '0.5rem', padding: '0.5rem', background: '#2a0000', borderRadius: '4px', fontSize: '0.8rem', color: '#ffaaaa' }}>
                                      ⚠️ <strong>WARNING:</strong> This command is restricted and may have dangerous side effects. Are you sure you want to proceed?
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      {message.pending_approval_steps && message.pending_approval_steps.length > 0 && (
                        <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem' }}>
                          <button
                            onClick={() => handleApprove(message.message_id, message.pending_approval_steps!)}
                            disabled={approving === message.message_id}
                            style={{
                              padding: '0.5rem 1rem',
                              background: approving === message.message_id ? '#444' : '#19c37d',
                              color: '#fff',
                              border: 'none',
                              borderRadius: '6px',
                              cursor: approving === message.message_id ? 'not-allowed' : 'pointer',
                              fontSize: '0.875rem',
                            }}
                          >
                            {approving === message.message_id ? 'Approving...' : '✓ Approve'}
                          </button>
                          <button
                            onClick={() => handleReject(message.message_id, message.pending_approval_steps!)}
                            disabled={approving === message.message_id}
                            style={{
                              padding: '0.5rem 1rem',
                              background: approving === message.message_id ? '#444' : '#dc3545',
                              color: '#fff',
                              border: 'none',
                              borderRadius: '6px',
                              cursor: approving === message.message_id ? 'not-allowed' : 'pointer',
                              fontSize: '0.875rem',
                            }}
                          >
                            {approving === message.message_id ? 'Rejecting...' : '✗ Reject'}
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
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
        )}
        </div>

        {/* Observability panel */}
        <ObservabilityPanel />
      </div>
    </div>
  );
}
