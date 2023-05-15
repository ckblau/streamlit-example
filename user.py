import torch

import utils


def use_model(data):
    model = torch.load('model_dict.pt')
    model.eval()
    data = utils.data_processed(data)
    pred = model(data)
    return pred


'''
data = {'Lot Frontage': -1, 'Lot Area': -1, 'Overall Qual': -1, 'Overall Cond': -1, 'Year Built': -1,
        'Year Remod/Add': -1, 'Mas Vnr Area': -1, 'BsmtFin SF 1': -1, 'Bsmt Unf SF': -1, 'Total Bsmt SF': -1,
        '1st Flr SF': -1, '2nd Flr SF': -1, 'Gr Liv Area': -1, 'Bsmt Full Bath': -1, 'Full Bath': -1, 'Half Bath': -1,
        'Bedroom AbvGr': -1, 'Fireplaces': -1, 'Garage Area': -1, 'Wood Deck SF': -1, 'Open Porch SF': -1,
        'Mo Sold': -1, 'Yr Sold': -1,
        'MS SubClass': '020', 'Sale Condition': 'Normal', 'Lot Shape': 'IR1', 'Lot Config': 'FR3',
        'Neighborhood': 'Blueste', 'Condition 1': 'Norm'}
print(use_model(data))
'''
