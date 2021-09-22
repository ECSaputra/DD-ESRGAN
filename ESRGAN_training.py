
### prep dataloaders

train_dataset = MyPatchDataset('DIV2K_train_HR/',64)
train_loader_args = dict(shuffle=True, batch_size=5, num_workers=4, pin_memory=True, drop_last=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_loader_args)

val_dataset = MyPatchDataset('DIV2K_valid_HR/',64)
val_loader_args = dict(shuffle=False, batch_size=5, num_workers=4, pin_memory=True, drop_last=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=1)
val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_loader_args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = GeneratorESRGAN(16, 0.2, 64, 32)
D = DiscriminatorESRGAN()

G.load_state_dict(checkpoint['state_dict_gen'])
D.load_state_dict(checkpoint['state_dict_dis'])

content_loss = vggloss(device)

lamb = 0.005
eta = 0.01
generator_loss = GeneratorLossESRGAN(content_loss, D, lamb, eta,device)

discriminator_loss = DiscriminatorLoss(D,device)

# metric
psnr = PSNR()

lrate = 1*10**(-4)
optimizerG = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=lrate, weight_decay=0)
schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG,'min',patience=2, factor=0.5)
optimizerD = torch.optim.Adam(D.parameters(), betas=(0.9, 0.999), lr=lrate, weight_decay=0)
schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD,'min',patience=2, factor=0.5)

### ESRGAN training

numEpochs = 1000

for epoch in range(numEpochs):
  print("="*40, " Epoch ", epoch+1)

  # start training
  avg_psnr= 0.0
  avg_psnr_val_epoch = 0.0


  print("Training...")
  printAvailSpace()
  for batch_num, (highres, lowres) in enumerate(train_dataloader):
      highres, lowres = highres.to(device), lowres.to(device)
          
      
      ### Train generator
      optimizerG.zero_grad()
      G.train()
              
      outputs = G(lowres)
      with torch.no_grad():
          avg_psnr += psnr(outputs, highres)

      gen_loss = generator_loss.forward(highres, outputs)
      gen_loss.backward()
      optimizerG.step()
      schedulerG.step(gen_loss)
      



      ### Train discriminator
      optimizerD.zero_grad()
      D.train()

      outputs.detach_()

      dis_loss = discriminator_loss.forward(highres, outputs)
      dis_loss.backward()
      optimizerD.step()
      schedulerD.step(dis_loss)

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
    highres, lowres = highres.to(device), lowres.to(device)

    outputs = G(lowres)

    with torch.no_grad():
        avg_psnr_val_epoch += psnr(outputs, highres)

  avg_psnr_val.append(avg_psnr_val_epoch/len(val_dataloader))


  for val_batch_num, (highres,lowres) in enumerate(val_dataloader):
    highres, lowres = highres.to(device), lowres.to(device)
    outputs = G(lowres)
    if val_batch_num == 1:
      break    

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
      print("-------- Saving checkpoint ----------")

      state = {
          'epoch': epoch,
          'state_dict_gen': G.state_dict(),
          'optimizer_gen': optimizerG.state_dict(),
          'state_dict_dis': D.state_dict(),
          'optimizer_dis':optimizerD.state_dict(),
          'train_psnr': avg_psnr_train[-1],
          'val_psnr': avg_psnr_val[-1]
      }
      torch.save(state, "best_model_largerPatch")