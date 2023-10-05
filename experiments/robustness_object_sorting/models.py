from seq2seq_abstracter_models import (Transformer, AutoregressiveSimpleAbstractor, Seq2SeqRelationalAbstracter, Seq2SeqSymbolicAbstracter, )


transformer_kwargs = dict(
    num_layers=4, num_heads=2, dff=64, 
    input_vocab='vector', embedding_dim=64)

rel_abstracter_kwargs = dict(
    num_layers=2, num_heads=2, dff=64, 
    input_vocab='vector', embedding_dim=64,
    rel_attention_activation='softmax'
    )
    
simpleabstractor_kwargs = dict(
    embedding_dim=64, input_vocab='vector',
    abstractor_kwargs=dict(num_layers=1, num_heads=8, dff=None,
        use_pos_embedding=False, mha_activation_type='softmax',
        attn_use_res=False, attn_use_layer_norm=True),
    decoder_kwargs=dict(num_layers=1, num_heads=4, dff=64, dropout_rate=0.1))

ablation_model_kwargs = dict(
    num_layers=2, num_heads=2, dff=64, 
    input_vocab='vector', embedding_dim=64,
    use_self_attn=True, use_encoder=True,
    mha_activation_type='softmax'
    )

def update_model_kwargs(seqs_length):
    for kwargs in [transformer_kwargs, rel_abstracter_kwargs, simpleabstractor_kwargs, ablation_model_kwargs]:
        kwargs['target_vocab'] = seqs_length + 1
        kwargs['output_dim'] = seqs_length

def get_model_kwargs(model_name):
    if model_name == 'transformer':
        return transformer_kwargs
    elif model_name == 'simple abstractor':
        return simpleabstractor_kwargs
    elif model_name == 'relational abstractor':
        return rel_abstracter_kwargs
    elif model_name == 'symbolic abstractor':
        return rel_abstracter_kwargs
    elif model_name == 'ablation model':
        return ablation_model_kwargs
    else:
        raise ValueError(f'`model_name` {model_name} is invalid')

def create_model(model_name):
    if model_name == 'transformer':
        return Transformer(**transformer_kwargs)
    elif model_name == 'simple abstractor':
        return AutoregressiveSimpleAbstractor(**simpleabstractor_kwargs)
    elif model_name == 'relational abstractor':
        return Seq2SeqRelationalAbstracter(**rel_abstracter_kwargs)
    elif model_name == 'symbolic abstractor':
        return Seq2SeqSymbolicAbstracter(**rel_abstracter_kwargs)
    elif model_name == 'ablation model':
        return AutoregressiveAblationAbstractor(**ablation_model_kwargs)
    else:
        raise ValueError(f'`model_name` {model_name} is invalid')
