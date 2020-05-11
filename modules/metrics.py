import torch
import torch.nn.functional as F
from pdb import set_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beam_search(learn, img,vocab = None, beam_size = 5):
    with torch.no_grad():
        k = beam_size
        
        ## imput tensor preparation
        img = img.unsqueeze(0) #treating as batch of size 1

        ## model prepartion
        mod = learn.model

        # encoder output
        encoder_out = mod.encoder(img)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)

        # expand or repeat 'k' time
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[vocab['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = mod.decoder.init_hidden_state(encoder_out)

        references = list()
        hypotheses = list()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = mod.decoder.embedding(k_prev_words).squeeze(1).float()  # (s, embed_dim)
            awe, _ = mod.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = mod.decoder.sigmoid(mod.decoder.f_beta(h))
            awe = (gate * awe)

            h, c = mod.decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = mod.decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            

            # Add scores to prev scores
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / len(vocab)  # (s)
            next_word_inds = top_k_words % len(vocab)  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1) stroes indices of words

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != vocab['<end>']]

            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)


            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    # Hypotheses
    hypotheses.append([w for w in seq if w not in {vocab['<start>'], vocab['<end>'], vocab['<pad>']}])

    return hypotheses
