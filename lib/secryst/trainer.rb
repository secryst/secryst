module Secryst
  class Trainer

    def initialize(
      model:,
      batch_size:,
      lr:,
      data_input:,
      data_target:,
      hyperparameters:,
      max_epochs: nil,
      log_interval: 10,
      checkpoint_every:15,
      checkpoint_dir:,
      scheduler_step_size:,
      gamma:
    )
      @data_input = File.readlines(data_input, chomp: true)
      @data_target = File.readlines(data_target, chomp: true)

      @device = "cpu"
      @lr = lr
      @scheduler_step_size = scheduler_step_size
      @gamma = gamma
      @batch_size = batch_size
      @model_name = model
      @max_epochs = max_epochs
      @log_interval = log_interval
      @checkpoint_every = checkpoint_every
      @checkpoint_dir = checkpoint_dir
      FileUtils.mkdir_p(@checkpoint_dir)
      last_checkpoint = Dir[File.join(@checkpoint_dir, '*')].sort_by {|n| n.scan(/checkpoint-([0-9]+)/)&.first&.first.to_i }.last
      puts "Starting from checkpoint #{last_checkpoint}" if last_checkpoint 
      @initial_epoch = last_checkpoint ? last_checkpoint.scan(/checkpoint-([0-9]+)/)&.first&.first.to_i + 1 : 0
      generate_vocabs_and_data

      @hyperparameters = hyperparameters.merge({
        input_vocab_size: @input_vocab.length,
        target_vocab_size: @target_vocab.length,
      })

      save_vocabs
      save_metadata

      case model
      when 'transformer'
        if last_checkpoint
          @model = Model.from_file(last_checkpoint)
        else
          @model = Secryst::Transformer.new(@hyperparameters)
        end
      else
        raise ArgumentError, 'Only transformer model is currently supported'
      end
    end

    def train
      best_model = nil
      best_val_loss = 1.0/0.0 # infinity

      return unless @model_name == 'transformer'

      criterion = Torch::NN::CrossEntropyLoss.new(ignore_index: index_of('<pad>')).to(@device)
      optimizer = Torch::Optim::SGD.new(@model.parameters, lr: @lr)
      scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: @scheduler_step_size, gamma: @gamma)

      total_loss = 0.0
      start_time = Time.now
      ntokens = @target_vocab.length
      epoch = @initial_epoch

      loop do
        epoch_start_time = Time.now
        @model.train
        @train_data.each.with_index do |batch, i|
          inputs, targets, decoder_inputs, src_mask, tgt_mask, memory_mask = batch
          inputs = Torch.tensor(inputs).t
          decoder_inputs = Torch.tensor(decoder_inputs).t
          targets = Torch.tensor(targets).t
          src_key_padding_mask = inputs.t.eq(1)
          tgt_key_padding_mask = decoder_inputs.t.eq(1)

          optimizer.zero_grad
          opts = {
            # src_mask: src_mask,
            tgt_mask: tgt_mask,
            # memory_mask: memory_mask,
            src_key_padding_mask: src_key_padding_mask,
            tgt_key_padding_mask: tgt_key_padding_mask,
            memory_key_padding_mask: src_key_padding_mask,
          }
          output = @model.call(inputs, decoder_inputs, opts)
          loss = criterion.call(output.transpose(0,1).reshape(-1, ntokens), targets.t.view(-1))
          loss.backward
          ClipGradNorm.clip_grad_norm(@model.parameters, max_norm: 0.5)
          optimizer.step

          # puts "i[#{i}] loss: #{loss}"
          total_loss += loss.item()
          if ( (i + 1) % @log_interval == 0 )
            cur_loss = total_loss / @log_interval
            elapsed = Time.now - start_time
            puts "| epoch #{epoch} | #{i + 1}/#{@train_data.length} batches | "\
                  "lr #{scheduler.get_lr()[0].round(4)} | ms/batch #{(1000*elapsed.to_f / @log_interval).round} | "\
                  "loss #{cur_loss.round(5)} | ppl #{Math.exp(cur_loss).round(5)}"
            total_loss = 0
            start_time = Time.now
          end
        end

        if epoch > 0 && epoch % @checkpoint_every == 0
          save_model(epoch)
        end

        # Evaluate
        @model.eval()
        total_loss = 0.0
        Torch.no_grad do
          @eval_data.each.with_index do |batch, i|
            inputs, targets, decoder_inputs, src_mask, tgt_mask, memory_mask = batch
            inputs = Torch.tensor(inputs).t
            decoder_inputs = Torch.tensor(decoder_inputs).t
            targets = Torch.tensor(targets).t
            src_key_padding_mask = inputs.t.eq(1)
            tgt_key_padding_mask = decoder_inputs.t.eq(1)

            opts = {
              # src_mask: src_mask,
              tgt_mask: tgt_mask,
              # memory_mask: memory_mask,
              src_key_padding_mask: src_key_padding_mask,
              tgt_key_padding_mask: tgt_key_padding_mask,
              memory_key_padding_mask: src_key_padding_mask,
            }
            output = @model.call(inputs, decoder_inputs, **opts)
            output_flat = output.transpose(0,1).reshape(-1, ntokens)

            total_loss += criterion.call(output_flat, targets.t.view(-1)).item
          end
          total_loss = total_loss / @eval_data.length
          puts('-' * 89)
          puts "| end of epoch #{epoch} | time: #{(Time.now - epoch_start_time).round(3)}s | "\
                  " valid loss #{total_loss.round(5)} | valid ppl #{Math.exp(total_loss).round(5)} "
          puts('-' * 89)
          if total_loss < best_val_loss
            best_model = @model
            best_val_loss = total_loss
          end
        end
        scheduler.step

        epoch += 1
        break if @max_epochs && (@initial_epoch + @max_epochs) < epoch
      end
    ensure
      cleanup_files
    end

    private

    def generate_vocabs_and_data
      input_texts = []
      target_texts = []
      input_vocab_counter = Hash.new(0)
      target_vocab_counter = Hash.new(0)

      @data_input.each do |input_text|
        input_text.strip!
        input_texts.push(input_text)
        input_text.each_char do |char|
          input_vocab_counter[char] += 1
        end
      end

      @data_target.each do |target_text|
        target_text.strip!
        target_texts.push(target_text)
        target_text.each_char do |char|
          target_vocab_counter[char] += 1
        end
      end

      @input_vocab = Vocab.new(input_vocab_counter.keys)
      @target_vocab = Vocab.new(target_vocab_counter.keys)

      # Generate train, eval, and test batches
      seed = 1
      zipped_texts = input_texts.zip(target_texts)
      zipped_texts = zipped_texts.shuffle(random: Random.new(seed))

      # train - 90%, eval - 7%, test - 3%
      train_texts = zipped_texts[0...(zipped_texts.length*0.9).to_i]
      eval_texts = zipped_texts[(zipped_texts.length*0.9).to_i + 1..(zipped_texts.length*0.97).to_i]
      test_texts = zipped_texts[(zipped_texts.length*0.97).to_i+1..-1]

      # prepare batches
      @train_data = batchify(train_texts)
      @eval_data = batchify(eval_texts)
      @test_data = batchify(test_texts)

    end

    def pad(arr, length, no_eos:false, no_sos:false)
      if !no_eos
        arr = arr + ["<eos>"]
      end
      if !no_sos
        arr = ["<sos>"] + arr
      end
      arr.fill("<pad>", arr.length...length)
    end

    def index_of(token)
      @target_vocab.stoi[token]
    end

    def batchify(data)
      batches = []
      # do loop at least one time
      [1,data.length / @batch_size].max.times do |i|
        input_data = data[i*@batch_size, @batch_size].transpose[0]
        decoder_input_data = data[i*@batch_size, @batch_size].transpose[1]
        target_data = data[i*@batch_size, @batch_size].transpose[1]
        max_input_seq_length = input_data.max_by(&:length).length + 2
        max_target_seq_length = target_data.max_by(&:length).length + 1
        src_mask = Torch.triu(Torch.ones(max_input_seq_length,max_input_seq_length)).eq(0).transpose(0,1)
        tgt_mask = Torch.triu(Torch.ones(max_target_seq_length,max_target_seq_length)).eq(0).transpose(0,1)
        memory_mask = Torch.triu(Torch.ones(max_input_seq_length,max_target_seq_length)).eq(0).transpose(0,1)
        batches << [
          input_data.map {|line| pad(line.chars, max_input_seq_length).map {|c| @input_vocab[c]} },
          target_data.map {|line| pad(line.chars, max_target_seq_length, no_sos: true).map {|c| @target_vocab[c]} },
          decoder_input_data.map {|line| pad(line.chars, max_target_seq_length, no_eos: true).map {|c| @target_vocab[c]} },
          src_mask,
          tgt_mask,
          memory_mask
        ]
      end

      batches
    end

    def save_vocabs
      File.write("#{@checkpoint_dir}/vocabs.yaml", {
        "input"  => @input_vocab.itos,
        "target" => @target_vocab.itos
      }.to_yaml)
    end

    def save_metadata
      File.write("#{@checkpoint_dir}/metadata.yaml", {
        "name"  => "transformer"
      }.merge(@hyperparameters).to_yaml)
    end

    def save_model(epoch)
      start_saving = Time.now
      Torch.save(@model.state_dict, "#{@checkpoint_dir}/model.pth")

      # Zip generation
      input_filenames = ['model.pth', 'metadata.yaml', 'vocabs.yaml']
      zipfile_name = "#{@checkpoint_dir}/checkpoint-#{epoch}.zip"
      FileUtils.rm(zipfile_name) if File.exists?(zipfile_name)
      Zip::File.open(zipfile_name, Zip::File::CREATE) do |zipfile|
        input_filenames.each do |filename|
          zipfile.add(filename, File.join(@checkpoint_dir, filename))
        end
      end

      puts ">> Saved checkpoint '#{@checkpoint_dir}/checkpoint-#{epoch}.zip' in #{(Time.now - start_saving).round(3)}s"
    end

    def cleanup_files
      FileUtils.rm("#{@checkpoint_dir}/model.pth") if File.exists?("#{@checkpoint_dir}/model.pth")
      FileUtils.rm("#{@checkpoint_dir}/metadata.yaml") if File.exists?("#{@checkpoint_dir}/metadata.yaml")
      FileUtils.rm("#{@checkpoint_dir}/vocabs.yaml") if File.exists?("#{@checkpoint_dir}/vocabs.yaml")
    end
  end
end