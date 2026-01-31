/**
 * IndexedDB message cache for chat application.
 * 
 * Schema:
 * - Object store: messages
 *   - messageId (primary key)
 *   - sessionId (indexed)
 *   - content
 *   - role
 *   - createdAt
 * - Indexes:
 *   - sessionId
 *   - sessionId + createdAt (compound)
 */

import type { MessageResponse } from '../types/api';

const DB_NAME = 'chat_cache';
const DB_VERSION = 1;
const STORE_NAME = 'messages';

interface CachedMessage {
  messageId: string;
  sessionId: string;
  content: string;
  role: string;
  createdAt: number; // Timestamp as number
}

let dbPromise: Promise<IDBDatabase> | null = null;

/**
 * Initialize IndexedDB database.
 */
function initDB(): Promise<IDBDatabase> {
  if (dbPromise) {
    return dbPromise;
  }

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Create object store if it doesn't exist
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, {
          keyPath: 'messageId',
        });

        // Create index on sessionId
        store.createIndex('sessionId', 'sessionId', { unique: false });

        // Create compound index for sessionId + createdAt (for sorting)
        store.createIndex('sessionId_createdAt', ['sessionId', 'createdAt'], {
          unique: false,
        });
      }
    };
  });

  return dbPromise;
}

/**
 * Get all messages for a session from cache.
 */
export async function getMessagesFromCache(
  sessionId: string
): Promise<MessageResponse[]> {
  try {
    const db = await initDB();
    const transaction = db.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const index = store.index('sessionId');
    const request = index.getAll(sessionId);

    return new Promise((resolve, reject) => {
      request.onsuccess = () => {
        const messages = request.result as CachedMessage[];
        // Sort by createdAt ascending
        messages.sort((a, b) => a.createdAt - b.createdAt);
        // Convert to MessageResponse format
        const result: MessageResponse[] = messages.map((msg) => ({
          message_id: msg.messageId,
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
          created_at: msg.createdAt,
        }));
        resolve(result);
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('Error getting messages from cache:', error);
    return [];
  }
}

/**
 * Save messages to cache (upsert - no duplicates).
 * Requires sessionId to be passed explicitly.
 */
export async function saveMessagesToCache(
  messages: MessageResponse[],
  sessionId: string
): Promise<void> {
  if (messages.length === 0) return;

  try {
    const db = await initDB();
    const transaction = db.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    // Upsert each message (messageId is primary key, so duplicates are automatically handled)
    const promises = messages.map((msg) => {
      const cached: CachedMessage = {
        messageId: msg.message_id,
        sessionId,
        content: msg.content,
        role: msg.role,
        createdAt: msg.created_at,
      };
      return store.put(cached);
    });

    await Promise.all(promises);
  } catch (error) {
    console.error('Error saving messages to cache:', error);
  }
}

/**
 * Save a single message to cache.
 */
export async function saveMessageToCache(
  message: MessageResponse,
  sessionId: string
): Promise<void> {
  try {
    const db = await initDB();
    const transaction = db.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    const cached: CachedMessage = {
      messageId: message.message_id,
      sessionId,
      content: message.content,
      role: message.role,
      createdAt: message.created_at,
    };

    await store.put(cached);
  } catch (error) {
    console.error('Error saving message to cache:', error);
  }
}

/**
 * Get the last message for a session (for delta sync cursor).
 */
export async function getLastMessageFromCache(
  sessionId: string
): Promise<MessageResponse | null> {
  try {
    const db = await initDB();
    const transaction = db.transaction([STORE_NAME], 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const index = store.index('sessionId_createdAt');
    
    // Get all messages for this session, sorted by createdAt
    const request = index.getAll([sessionId]);
    
    return new Promise((resolve, reject) => {
      request.onsuccess = () => {
        const messages = request.result as CachedMessage[];
        if (messages.length === 0) {
          resolve(null);
          return;
        }
        
        // Get the last message (highest createdAt)
        const lastMessage = messages[messages.length - 1];
        resolve({
          message_id: lastMessage.messageId,
          role: lastMessage.role as 'user' | 'assistant',
          content: lastMessage.content,
          created_at: lastMessage.createdAt,
        });
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('Error getting last message from cache:', error);
    return null;
  }
}

/**
 * Delete all messages for a session from cache.
 */
export async function deleteMessagesFromCache(sessionId: string): Promise<void> {
  try {
    const db = await initDB();
    const transaction = db.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const index = store.index('sessionId');
    const request = index.openCursor(IDBKeyRange.only(sessionId));

    return new Promise((resolve, reject) => {
      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        } else {
          resolve();
        }
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('Error deleting messages from cache:', error);
  }
}


/**
 * Check if cache has messages for a session.
 */
export async function hasCachedMessages(sessionId: string): Promise<boolean> {
  try {
    const messages = await getMessagesFromCache(sessionId);
    return messages.length > 0;
  } catch (error) {
    return false;
  }
}
