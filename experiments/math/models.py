import sys; sys.path += ['..', '../..']
from autoregressive_abstractor import AutoregressiveAbstractor
from seq2seq_abstracter_models import Transformer
# from tfm_transformer import TFMTransformer
# from tfm_abstractor import TFMAutoregressiveAbstractor, TFMAutoregressiveCompisitionalAbstractor

#region common kwargs

d_model = 512
num_heads = 8
dff = 2048
num_layers = 1

def get_params_by_size(size):
    if size=='small':
        d_model, num_heads, dff, num_layers = (64, 2, 128, 1)
    elif size=='medium':
        d_model, num_heads, dff, num_layers = (128, 4, 256, 1)
    elif size=='medium+':
        d_model, num_heads, dff, num_layers = (200, 4, 400, 1)
    elif size=='medium++':
        d_model, num_heads, dff, num_layers = (256, 4, 512, 1)
    elif size=='large':
        d_model, num_heads, dff, num_layers = (256, 8, 1024, 2)
    elif size=='x-large':
        d_model, num_heads, dff, num_layers = (512, 8, 2048, 2)
    else:
        raise ValueError(f'size {size} invalid')

    return d_model, num_heads, dff, num_layers
#endregion

#region Transformer
def create_transformer(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    transformer = Transformer(
        num_layers=num_layers, num_heads=num_heads, dff=dff, embedding_dim=d_model,
        input_vocab=input_vocab_size, target_vocab=target_vocab_size,
        output_dim=target_vocab_size, dropout_rate=0.1,)

    return transformer
#endregion

#region TFM-Transformer
def create_tfmtransformer(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    transformer = TFMTransformer(
        num_layers=num_layers, num_heads=num_heads, dff=dff, embedding_dim=d_model,
        input_vocab=input_vocab_size, target_vocab=target_vocab_size,
        output_dim=target_vocab_size, dropout_rate=0.1,)

    return transformer
#endregion


#region Abstractor
def create_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, rel_dim=num_heads, dff=dff, symbol_dim=d_model,
        proj_dim=d_model//num_heads, symmetric_rels=False, encoder_kwargs=None,
        rel_activation_type='softmax', use_self_attn=False, use_layer_norm=False,
        dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='abstractor', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region Abstractor
def create_abstractor2(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, rel_dim=num_heads, dff=dff, symbol_dim=d_model,
        proj_dim=d_model//num_heads, symmetric_rels=False, encoder_kwargs=None,
        rel_activation_type='tanh', use_self_attn=True, use_layer_norm=False,
        dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='abstractor', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region RelationalAbstractor
def create_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=True)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region RelationalAbstractor
def create_linear_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='linear', use_self_attn=False)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region RelationalAbstractor
def create_relational_abstractor2(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=True)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor
def create_symbolretrieving_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=num_heads, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SyntacticAbstractor
def create_syntactic_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=1, num_heads=num_heads, dff=dff, n_symbols=16,
        symbol_n_heads=1, symbol_binding_dim=None, add_pos_embedding=True, symbol_retriever_type=1)

    model = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='syntactic',
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

#region SymbolRetrievingAbstractor
def create_symbolretrieving_abstractor_archd(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=num_heads, add_pos_embedding=True, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor
def create_symbolretrieving_abstractor_archd_fewsymbols(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=16, symbol_n_heads=1, add_pos_embedding=True, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor
def create_symbolretrieving2_abstractor_archd(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=num_heads, add_pos_embedding=True, symbol_retriever_type=2,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor
def create_symbolretrieving2_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=num_heads, symbol_retriever_type=2,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor
def create_symbolretrieving_singlehead_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=1, symbol_retriever_type=1,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

#region SymbolRetrievingAbstractor
def create_symbolretrieving2_singlehead_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)

    encoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=1, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, num_heads=num_heads, dff=dff,
        n_symbols=256, symbol_n_heads=1, symbol_retriever_type=2,
        rel_activation_function='softmax', use_self_attn=True, dropout_rate=0.1)

    abstractor = AutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='symbol-retrieving', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')
    return abstractor
#endregion

# TODO: what to do with tfm implementations? remove? delete?

#region TFM-Abstractor
def create_tfm_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_attention_heads=num_heads, intermediate_size=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=num_layers, rel_dim=num_heads, dff=dff, symbol_dim=d_model,
        proj_dim=d_model//num_heads, symmetric_rels=False, use_learned_symbols=False,
        rel_activation_type='linear', use_self_attn=False, use_layer_norm=False,
        dropout_rate=0.1)
    # abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
    #     use_learned_symbols=False, mha_activation_type='linear', use_self_attn=False)


    abstractor = TFMAutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='abstractor', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

#region TFM-Abstractor
def create_tfm_compisitional_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_attention_heads=num_heads, intermediate_size=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(
        num_layers=1, rel_dim=num_heads, dff=dff, symbol_dim=d_model,
        proj_dim=d_model//num_heads, symmetric_rels=False, use_learned_symbols=False,
        rel_activation_type='linear', use_self_attn=False, use_layer_norm=False,
        dropout_rate=0.1)

    abstractor = TFMAutoregressiveCompisitionalAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='abstractor', # 'abstractor', 'simple', 'relational', or 'symbolic'
        n_abstractors=num_layers,
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

#region TFM-Abstractor
def create_tfm_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_attention_heads=num_heads, intermediate_size=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=False)

    abstractor = TFMAutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

#region TFM-Abstractor
def create_tfm_1layer_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_attention_heads=num_heads, intermediate_size=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=1, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=False)

    abstractor = TFMAutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

#region TFM-Abstractor
def create_tfm_relational_abstractor2(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_attention_heads=num_heads, intermediate_size=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='softmax', use_self_attn=False)

    abstractor = TFMAutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='input', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

#region TFM-Abstractor
def create_tfm_linear_relational_abstractor(input_vocab_size, target_vocab_size, size='x-large'):
    d_model, num_heads, dff, num_layers = get_params_by_size(size)
    encoder_kwargs = dict(num_layers=num_layers, num_attention_heads=num_heads, intermediate_size=dff, dropout_rate=0.1,)
    decoder_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff, dropout_rate=0.1,)
    abstractor_kwargs = dict(num_layers=num_layers, num_heads=num_heads, dff=dff,
        use_learned_symbols=False, mha_activation_type='linear', use_self_attn=False)

    abstractor = TFMAutoregressiveAbstractor(
        encoder_kwargs,
        abstractor_kwargs,
        decoder_kwargs,
        input_vocab=input_vocab_size,
        target_vocab=target_vocab_size,
        embedding_dim=d_model,
        output_dim=target_vocab_size,
        abstractor_type='relational', # 'abstractor', 'simple', 'relational', or 'symbolic'
        abstractor_on='encoder', # 'input' or 'encoder'
        decoder_on='encoder-abstractor', # 'abstractor' or 'encoder-abstractor'
        name='autoregressive_abstractor')

    return abstractor
#endregion

model_creator_dict = dict(
    transformer=create_transformer,
    tfm_transformer=create_tfmtransformer,
    abstractor=create_abstractor,
    tfm_abstractor=create_tfm_abstractor,
    tfm_relational_abstractor=create_tfm_relational_abstractor,
    tfm_1layer_relational_abstractor=create_tfm_1layer_relational_abstractor,
    tfm_relational_abstractor2=create_tfm_relational_abstractor2,
    tfm_linear_relational_abstractor=create_tfm_relational_abstractor,
    tfm_compisitional_abstractor=create_tfm_compisitional_abstractor,
    relational_abstractor=create_relational_abstractor,
    syntactic_abstractor=create_syntactic_abstractor,
    symbolretrieving_abstractor=create_symbolretrieving_abstractor,
    symbolretrieving2_abstractor=create_symbolretrieving2_abstractor,
    symbolretrieving_singlehead_abstractor=create_symbolretrieving_singlehead_abstractor,
    symbolretrieving2_singlehead_abstractor=create_symbolretrieving2_singlehead_abstractor,
    symbolretrieving_abstractor_archd=create_symbolretrieving_abstractor_archd,
    symbolretrieving_abstractor_archd_fewsymbols=create_symbolretrieving_abstractor_archd_fewsymbols,
    symbolretrieving2_abstractor_archd=create_symbolretrieving2_abstractor_archd,
    linear_relational_abstractor=create_linear_relational_abstractor,
    relational_abstractor2=create_relational_abstractor2
    )
