module Secryst
  class Vocab
    UNK = "<unk>"
    attr_reader :stoi, :itos

    def initialize(
      list, specials: ["<unk>", "<pad>", "<sos>", "<eos>"], specials_first: true
    )
      @unk_index = nil
      @itos = []
      if specials_first && (list & specials).length == 0
        @itos = specials
      end

      @itos += list

      if specials.include?(UNK)  # hard-coded for now
        unk_index = specials.index(UNK)  # position in list
        # account for ordering of specials, set variable
        @unk_index = specials_first ? unk_index : @itos.length + unk_index
        @stoi = Hash.new(@unk_index)
      else
        @stoi = {}
      end

      if !specials_first && (list & specials).length == 0
        @itos.concat(specials)
      end

      # stoi is simply a reverse dict for itos
      @itos.each_with_index do |tok, i|
        @stoi[tok] = i
      end
    end

    def [](token)
      @stoi.fetch(token, @stoi.fetch(UNK))
    end

    def length
      @itos.length
    end
    alias_method :size, :length
  end
end