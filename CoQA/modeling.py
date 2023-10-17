from bert import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import random
import torch
from run_coqa_dataset_utils import read_one_coqa_example_extern,convert_one_example_to_features,recover_predicted_answer,RawResult
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import pdb

class Multi_linear_layer(nn.Module):
    def __init__(self,
                 n_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 activation=None):
        super(Multi_linear_layer, self).__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_size, hidden_size))
        for _ in range(1, n_layers - 1):
            self.linears.append(nn.Linear(hidden_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, output_size))
        self.activation = getattr(F, activation)

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        linear = self.linears[-1]
        x = linear(x)
        return x

class BertForCoQA(BertPreTrainedModel):
    def __init__(
            self,
            config,
            output_attentions=False,
            keep_multihead_output=False,
            n_layers=2,
            activation='relu',
            beta=100,
            device='cuda'
    ):
        super(BertForCoQA, self).__init__(config)
        self.output_attentions = output_attentions
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.rational_l = Multi_linear_layer(n_layers, hidden_size,
                                             hidden_size, 1, activation)
        self.logits_l = Multi_linear_layer(n_layers, hidden_size, hidden_size,
                                           2, activation)
        self.unk_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 1,
                                        activation)
        self.attention_l = Multi_linear_layer(n_layers, hidden_size,
                                              hidden_size, 1, activation)
        self.yn_l = Multi_linear_layer(n_layers, hidden_size, hidden_size, 2,
                                       activation)
        self.beta = beta

        self.init_weights()
        
        self.device = device

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            rational_mask=None,
            cls_idx = None,
            head_mask=None,
    ):
        # mask some words on inputs_ids
        # if self.training and self.mask_p > 0:
        #     batch_size = input_ids.size(0)
        #     for i in range(batch_size):
        #         len_c, len_qc = token_type_ids[i].sum(
        #             dim=0).detach().item(), attention_mask[i].sum(
        #                 dim=0).detach().item()
        #         masked_idx = random.sample(range(len_qc - len_c, len_qc),
        #                                    int(len_c * self.mask_p))
        #         input_ids[i, masked_idx] = 100

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            # output_all_encoded_layers=False,
            head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, sequence_output, cls_outputs = outputs
        else:
            final_hidden, pooled_output = outputs

        rational_logits = self.rational_l(final_hidden)
        rational_logits = torch.sigmoid(rational_logits)

        final_hidden = final_hidden * rational_logits

        logits = self.logits_l(final_hidden)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits, end_logits = start_logits.squeeze(
            -1), end_logits.squeeze(-1)

        segment_mask = token_type_ids.type(final_hidden.dtype)

        rational_logits = rational_logits.squeeze(-1) * segment_mask

        start_logits = start_logits * rational_logits

        end_logits = end_logits * rational_logits

        unk_logits = self.unk_l(pooled_output)

        attention = self.attention_l(final_hidden).squeeze(-1)

        attention.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        attention = F.softmax(attention, dim=-1)

        attention_pooled_output = (attention.unsqueeze(-1) *
                                   final_hidden).sum(dim=-2)

        yn_logits = self.yn_l(attention_pooled_output)

        yes_logits, no_logits = yn_logits.split(1, dim=-1)

        start_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        end_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        new_start_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, start_logits), dim=-1)
        new_end_logits = torch.cat(
            (yes_logits, no_logits, unk_logits, end_logits), dim=-1)

        if start_positions is not None and end_positions is not None:

            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)

            # rational part
            alpha = 0.25
            gamma = 2.
            rational_mask = rational_mask.type(final_hidden.dtype)

            rational_loss = -alpha * (
                (1 - rational_logits)**gamma
            ) * rational_mask * torch.log(rational_logits + 1e-7) - (
                1 - alpha) * (rational_logits**gamma) * (
                    1 - rational_mask) * torch.log(1 - rational_logits + 1e-7)

            rational_loss = (rational_loss *
                             segment_mask).sum() / segment_mask.sum()
            # end

            assert not torch.isnan(rational_loss)

            total_loss = (start_loss +
                          end_loss) / 2 + rational_loss * self.beta
            return total_loss

        return start_logits, end_logits, yes_logits, no_logits, unk_logits




    def predict_one_automatic_turn(self,partial_example,unique_id, example_idx, tokenizer):
        question = partial_example.question_text
        # pdb.set_trace()
        turn = int(partial_example.qas_id.split("#")[1])
        example = read_one_coqa_example_extern(partial_example, self.QA_history, history_len=2, add_QA_tag=False)
        
        curr_eval_features, next_unique_id= convert_one_example_to_features(example=example, unique_id=unique_id, example_index=example_idx, tokenizer=tokenizer, max_seq_length=512,
                                    doc_stride=128, max_query_length=64)
        all_input_ids = torch.tensor([f.input_ids for f in curr_eval_features],
                                            dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in curr_eval_features],
                                    dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in curr_eval_features],
                                    dtype=torch.long)
        all_feature_index = torch.arange(all_input_ids.size(0),
                                        dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                all_segment_ids, all_feature_index)
        # Run prediction for full data

        eval_dataloader = DataLoader(eval_data,
                                    sampler=None,
                                    batch_size=1)
        curr_results = []
        # Run prediction for current example
        for input_ids, input_mask, segment_ids, feature_indices in eval_dataloader:

            input_ids = input_ids.to(self.device)
            
            ##############################
            # print('>>>>>>>>input_ids>>>>>>>>')
            # print(tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy().tolist()))
            # print(tokenizer.convert_ids_to_tokens(input_ids))
            # pdb.set_trace()
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            print(type(input_ids[0]), type(input_mask[0]), type(segment_ids[0]))
            # Assume the logits are a list of one item
            with torch.no_grad():
                batch_start_logits, batch_end_logits, batch_yes_logits, batch_no_logits, batch_unk_logits = self.forward(
                    input_ids, segment_ids, input_mask)
            for i, feature_index in enumerate(feature_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                yes_logits = batch_yes_logits[i].detach().cpu().tolist()
                no_logits = batch_no_logits[i].detach().cpu().tolist()
                unk_logits = batch_unk_logits[i].detach().cpu().tolist()
                eval_feature = curr_eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                curr_results.append(
                    RawResult(unique_id=unique_id,
                                start_logits=start_logits,
                                end_logits=end_logits,
                                yes_logits=yes_logits,
                                no_logits=no_logits,
                                unk_logits=unk_logits))
        predicted_answer = recover_predicted_answer(
            example=example, features=curr_eval_features, results=curr_results, tokenizer=tokenizer, n_best_size=20, max_answer_length=30,
            do_lower_case=True,verbose_logging=False)
        # pdb.set_trace()
        self.QA_history.append((turn, question, (predicted_answer, None, None)))
        return predicted_answer, next_unique_id

