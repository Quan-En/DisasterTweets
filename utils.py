import re

from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, token_tensor, mask_tensor, target_tensor=None):
      self.token_tensor = token_tensor
      self.mask_tensor = mask_tensor
      self.target_tensor = target_tensor

    def __getitem__(self, index):
      if self.target_tensor != None: return self.token_tensor[index], self.mask_tensor[index], self.target_tensor[index]
      else: return self.token_tensor[index], self.mask_tensor[index]
    
    def __len__(self):
        return self.token_tensor.size(0)

def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove trailing whitespace
    """
    s = s.lower()
    # Remove url
    s = re.sub(r"https?://[a-zA-Z0-9+-=\\/@[\];:,.!^'\"#$%&()~|`{}*<>?_]+", "", s)
    # Remove html
    s = re.sub(r"<(\".*?\"|\'.*?\'|[^\'\"])*?>", "", s)
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s