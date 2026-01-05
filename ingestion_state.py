import json
import os
import hashlib
from datetime import datetime

class IngestionStateManager:
    def __init__(self, state_file_path):
        self.state_file_path = state_file_path
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load ingestion state: {e}. Starting fresh.")
                return {}
        return {}

    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"Error saving ingestion state: {e}")

    def get_url_state(self, url):
        return self.state.get(url, {})

    def update_url_state(self, url, last_modified=None, etag=None, content_hash=None):
        if url not in self.state:
            self.state[url] = {}
        
        entry = self.state[url]
        entry['last_checked'] = datetime.now().isoformat()
        if last_modified:
            entry['last_modified'] = last_modified
        if etag:
            entry['etag'] = etag
        if content_hash:
            entry['content_hash'] = content_hash
        
        self.save_state()

    def is_modified(self, url, new_last_modified=None, new_etag=None, new_content_hash=None):
        """
        Determines if the content has changed based on available metadata.
        Returns True if changed or unknown, False if strictly not changed.
        """
        stored = self.state.get(url)
        if not stored:
            return True # New URL
        
        # 1. Check ETag (Strong validator)
        if new_etag and stored.get('etag'):
            if new_etag != stored['etag']:
                return True
            else:
                return False # ETag matches, definitely not modified

        # 2. Check Last-Modified (Weak validator)
        if new_last_modified and stored.get('last_modified'):
             # String comparison might be enough if format is identical, 
             # but ideally we parse. For this simplified version, strict string equality.
             if new_last_modified != stored['last_modified']:
                 return True
             else:
                 return False

        # 3. Content Hash (Fallback if we downloaded content to check)
        if new_content_hash and stored.get('content_hash'):
            if new_content_hash != stored['content_hash']:
                return True
            else:
                return False

        # Default fallback: If we have no new info to compare, assume we need to check deeper 
        # (but caller usually calls this with at least one new value).
        # If we are here, it means we didn't have matching keys to compare.
        return True 
