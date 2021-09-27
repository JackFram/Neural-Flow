net = load_state_dict(path)
net_interpreter = ResNetInt(net)
train_batch = get_dataloader
feature_list, name_list, out = net_interpreter(train_batch)

visualize(feature_list, name_list)

