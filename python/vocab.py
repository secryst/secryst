class Vocab:
  _UNK_ = "<unk>"
  # attr_reader: stoi, : itos

  def __init__(self, vocab, specials=None, specials_first=True):
    if specials is None:
      specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
      
    self.unk_index = None
    self.itos = []
    # Here is a bit not precise conversion from ruby to py
    if specials_first and len(set(vocab) & set(specials)) == 0:
      self.itos = specials

    self.itos += vocab
    if not specials_first and len(set(vocab) & set(specials)) == 0:
      self.itos += specials

    # # Automatic substitution of unknown symbols
    # if "<unk>" in self.itos:
    #   unk_index = self.itos.index("<unk>")
    #   self.stoi = Hash.new(unk_index)
    # elsif self.itos.include?("[UNK]")
    #   unk_index = self.itos.index("[UNK]")
    #   self.stoi = Hash.new(unk_index)
    # else
    # end
    self.stoi = {}

    # stoi is simply a reverse dict for itos
    for i,tok in enumerate(self.itos):
      self.stoi[tok] = i

  def __getitem__(self, token):
    return self.stoi.get(token, self.stoi.get(self._UNK_))

  def __len__(self):
    return len(self.itos)
