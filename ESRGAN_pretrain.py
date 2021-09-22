
### prep dataloaders

train_dataset = MyPatchDataset('DIV2K_train_HR/',64)
train_loader_args = dict(shuffle=True, batch_size=5, num_workers=4, pin_memory=True, drop_last=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_loader_args)

val_dataset = MyPatchDataset('DIV2K_valid_HR/',64)
val_loader_args = dict(shuffle=False, batch_size=5, num_workers=4, pin_memory=True, drop_last=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=1)
val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_loader_args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = GeneratorESRGAN(16, 0.2, 64, 32)
G.load_state_dict(checkpoint['state_dict_gen'])

criterion= nn.L1Loss()

# metric
psnr = PSNR()

### PSNR-oriented training with L1 loss

lrate = 2*10**(-4)
optimizerG = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=lrate, weight_decay=0)
schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG,'min',patience=2, factor=0.5)

numEpochs = 300
start_time = time.time()
avg_psnr_train = []
avg_psnr_val = []

for epoch in range(numEpochs):
  print("="*40, " Epoch ", epoch+1)

  # start training
  avg_psnr= 0.0
  avg_psnr_val_epoch = 0.0


  print("Training...")
  printAvailSpace()
  for batch_num, (highres, lowres) in enumerate(train_dataloader):
      highres, lowres = highres.to(device), lowres.to(device)
          
      optimizerG.zero_grad()
      G.train()
              
      for i in range(5):
          outputs = G(lowres)

          loss = criterion(outputs,highres)
          loss.backward()
          optimizerG.step()
          schedulerG.step(loss)
          
          with torch.no_grad():
              avg_psnr += psnr(outputs, highres)

      del highres
      del lowres
      del outputs
      torch.cuda.empty_cache()
      torch.no_grad()



  avg_psnr_train.append(avg_psnr/len(train_dataloader))
  
  print("Training done.")
  printAvailSpace()

  print("Validating...")
  G.eval()
  for val_batch_num, (highres,lowres) in enumerate(val_dataloader):
#    print("batch ", val_batch_num)
#    printAvailSpace()
    highres, lowres = highres.to(device), lowres.to(device)

    outputs = G(lowres)


    with torch.no_grad():
        avg_psnr_val_epoch += psnr(outputs, highres)

  avg_psnr_val.append(avg_psnr_val_epoch/len(val_dataloader))
  
  plt.clf()
  plt.subplot(1, 3, 1)
  plt.imshow(lowres[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
  plt.subplot(1, 3, 2)
  plt.imshow(outputs[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
  plt.subplot(1, 3, 3)
  plt.imshow(highres[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
  plt.show()

  del outputs
  del highres
  del lowres
  torch.cuda.empty_cache()
  torch.no_grad()

  
  print(batch_num, ", Train PSNR: ", avg_psnr_train[-1], ", Val PSNR: ", avg_psnr_val[-1]) #avg_psnr/(batch_num+1))

  if avg_psnr_val[-1] >= max(avg_psnr_val):
      print("saving model")
      best_weightsG = G.state_dict()
      torch.save(best_weightsG, "./best_weight_Generator.model")
