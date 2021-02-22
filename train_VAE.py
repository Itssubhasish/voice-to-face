# set the learning parameters
lr = 0.001
epochs = 100
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

# dataset and dataloader
print('Parsing your dataset...')
voice_list, _, _ = get_dataset()
#####################################################
#NETWORKS_PARAMETERS['c']['output_channel'] = 60 ### Need to check once

print('Preparing the datasets...')
voice_dataset = VoiceDataset(voice_list,[300, 800])
#face_dataset = FaceDataset(face_list)

print('Dataset preparation Complete. Preparing the dataloaders...')
collate_fn = get_collate_fn([300, 800])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size= 128,
                          num_workers= 1,
                          collate_fn=collate_fn)
"""face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size= 128,
                         num_workers= 1)"""
#v_net, v_optimizer = get_network_voice(train = True)

voice_iterator = iter(cycle(voice_loader))
"""face_iterator = iter(cycle(face_loader))"""



train_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, voice_loader, voice_dataset, device, optimizer, criterion
    )
    """valid_epoch_loss, recon_images = validate(
        model, testloader, testset, device, criterion
    )"""
    train_loss.append(train_epoch_loss)
    #valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    #save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    #image_grid = make_grid(recon_images.detach().cpu())
    #grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    #print(f"Val Loss: {valid_epoch_loss:.4f}")