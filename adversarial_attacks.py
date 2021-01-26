import time
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients

from utils import *


""""""""""""""""""" FAST GRADIENT SIGN METHOD """""""""""""""""""

def fgsm(model, image, label, output, epsilon, clip=True, dataset='cifar10'):
  # Calculate the loss
  loss = F.nll_loss(output, label)
  # Zero all existing gradients
  model.zero_grad()
  # Calculate gradients of model in backward pass
  loss.backward()
  # Collect datagrad
  data_grad = image.grad.data
  # Collect the element-wise sign of the data gradient
  sign_data_grad = data_grad.sign()
  # Create the perturbation (considering data normalization)
  std = std_cifar10 if dataset=='cifar10' else std_ImageNet
  adv_pert = sign_data_grad
  adv_pert[0][0] = adv_pert[0][0] * (epsilon / std[0])
  adv_pert[0][1] = adv_pert[0][1] * (epsilon / std[1])
  adv_pert[0][2] = adv_pert[0][2] * (epsilon / std[2])
  # Create the perturbed image by adjusting each pixel of the input image
  perturbed_image = image + adv_pert
  # Adding clipping to maintain [0,1] range
  if clip:
    perturbed_image = clamp(perturbed_image, 0, 1, dataset)
  # Return the perturbed image and the perturbation
  return perturbed_image, adv_pert


""""""""""""""""""""""""""" DEEPFOOL """""""""""""""""""""""""""

def deepfool(model, device, im, num_classes=10, overshoot=0.02, lambda_fac=1.01, max_iter=50, p=2, clip=False, dataset='cifar10'):

  image = copy.deepcopy(im)

  # Get the input image shape
  input_shape = image.size()

  # Get the output of the original image
  output = model(image)

  # Array with the class probabilities of the image
  f_image = output.data.cpu().numpy().flatten()
  # Classes ordered by probability (descending)
  I = f_image.argsort()[::-1]
  # We consider only 'num_classes' classes
  I = I[0:num_classes]

  # Get the predicted label
  label = I[0]

  # Start from a copy of the original image
  pert_image = copy.deepcopy(image)   # tensor of size (1,3,H,W)

  # Initialize variables
  r_tot = torch.zeros(input_shape).to(device) # adversarial perturbation
  k_i = label  # current label
  loop_i = 0

  while k_i == label and loop_i < max_iter:

    # Get the output for the current image
    x = pert_image.clone().detach().requires_grad_(True)
    fs = model(x)

    pert = torch.Tensor([np.inf])[0].to(device)
    w = torch.zeros(input_shape).to(device)

    # Calculate grad(f_label(x_i))
    fs[0, I[0]].backward(retain_graph=True)
    grad_orig = copy.deepcopy(x.grad.data)

    for k in range(1, num_classes):  # for k != label
      # Reset gradients
      zero_gradients(x)

      # Calculate grad(f_k(x_i))
      fs[0, I[k]].backward(retain_graph=True)
      cur_grad = copy.deepcopy(x.grad.data)

      # Set new w_k and new f_k
      w_k = cur_grad - grad_orig
      f_k = (fs[0, I[k]] - fs[0, I[0]]).data

      # Calculate hyperplane-k distance
      if p == 2:
        pert_k = torch.abs(f_k) / w_k.norm()  # Frobenious norm (2-norm)
      elif p == np.inf:
        pert_k = torch.abs(f_k) / w_k.norm(1) # 1-norm

      # determine which w_k to use
      if pert_k < pert:
        pert = pert_k + 0.
        w = w_k + 0.

    # compute r_i and r_tot
    if p == 2:
      r_i = torch.clamp(pert, min=1e-4) * w / w.norm()  # Added 1e-4 for numerical stability
    elif p == np.inf:
      r_i = torch.clamp(pert, min=1e-4) * torch.sign(w)

    r_tot = r_tot + r_i

    # Update perturbed image
    pert_image = pert_image + r_i  # x_(i+1) <- x_i + r_i

    # Adding overshoot
    check_fool = image + (1 + overshoot) * r_tot

    x = check_fool.clone().detach().requires_grad_(True)
    # output for x_(i+1)
    fs = model(x)
    # label assigned to x_(i+1)
    k_i = torch.argmax(fs.data).item()

    loop_i += 1

  # Compute final perturbed image output
  x = pert_image.clone().detach().requires_grad_(True)
  fs = model(x)
  # Compute final gradient
  (fs[0, k_i] - fs[0, label]).backward(retain_graph=True)
  grad = copy.deepcopy(x.grad.data)
  grad = grad / grad.norm()

  # Include lambda_fac in the adversarial perturbation
  r_tot = lambda_fac * r_tot  # for SparseFool

  # Adding clipping to maintain [0,1] range
  if clip:
    pert_image = clamp(image + r_tot, 0, 1, dataset)

  else:
    pert_image = (image + r_tot).clone().detach()

  return grad, pert_image, r_tot, loop_i




"""""""""""""""" TEST A METHOD IN A SINGLE IMAGE """""""""""""""

# Test the desired method in one image
def test_method(model, device, img, label, method, params, show_pert=True):

  img = img.clone()

  model = model.to(device).eval()

  x = img.to(device)
  label = label.to(device)

  x.requires_grad = True

  y = model(x)
  init_pred = y.max(1, keepdim=True)[1]
  x_conf = F.softmax(y, dim=1).max(1, keepdim=True)[0].item()

  if init_pred.item() != label.item():
    print("Wrong classification...")
    return

  # Call method
  if method == 'fgsm':
    adv_x, pert_x = fgsm(model, x, label, y, params["epsilon"], params["clip"])

  elif method == 'deepfool':
    _, adv_x, pert_x, n_iter = deepfool(model, device, x, params["num_classes"], overshoot=params["overshoot"], max_iter=params["max_iter"], p=params["p"], clip=params["clip"])

  y_adv = model(adv_x)
  adv_pred = y_adv.max(1, keepdim=True)[1]
  adv_x_conf = F.softmax(y_adv, dim=1).max(1, keepdim=True)[0].item()

  # Plot results
  if show_pert:
    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.title('Original image')
    plt.axis('off')
    f.text(.25, .3, cifar10_classes[label.item()] + ' ({:.2f}%)'.format(x_conf*100), ha='center')
    plt.imshow(displayable(img))
    f.add_subplot(1,3,2)
    plt.title('Perturbation')
    plt.axis('off')
    plt.imshow(displayable(pert_x.cpu().detach()))
    f.add_subplot(1,3,3)
    plt.title('Adv. image')
    plt.axis('off')
    f.text(.8, .3, cifar10_classes[adv_pred.item()] + ' ({:.2f}%)'.format(adv_x_conf*100), ha='center')
    plt.imshow(displayable(adv_x.cpu().detach()))
    plt.show(block=True)

  else:
    f = plt.figure()
    plt.axis('off')
    f.text(.51, .08, cifar10_classes[adv_pred.item()] + ' ({:.2f}%)'.format(adv_x_conf*100), ha='center', fontsize=15)
    plt.imshow(displayable(adv_x.cpu().detach()))
    plt.show(block=True)

  if method == 'deepfool':
    print('Number of iterations needed: ', n_iter)

  return adv_pred.item() == label.item(), x_conf*100



""""""""" PERFORM A COMPLETE ATTACK ON CIFAR10 TEST SET """""""""

# Performs an attack and shows the results achieved by some method
def attack_model(model, device, test_loader, method, params, p=2, iters=10000, dataset='cifar10'):

  # Initialize the network and set the model in evaluation mode.
  model = model.to(device).eval()

  # Initialize stat counters
  correct = 0
  incorrect = 0
  total_time = 0
  method_iters = 0

  i = 0

  # Loop (iters) examples in test set
  for data, target in pbar(test_loader):
    if i >= iters:
      break
    i += 1

    # Send the data and label to the device
    data, target = data.to(device), target.to(device)

    # Set requires_grad attribute of tensor
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # If the initial prediction is wrong, dont botter attacking
    if init_pred.item() != target.item():
      continue

    if method == 'fgsm':
        # Call FGSM attack
        time_ini = time.time()
        perturbed_data, _ = fgsm(model, data, target, output, params["epsilon"], params["clip"], dataset)
        time_end = time.time()
        total_time += time_end-time_ini

    elif method == 'deepfool':
        # Call DeepFool attack
        time_ini = time.time()
        _, perturbed_data, _, n_iter = deepfool(model, device, data, params["num_classes"], overshoot=params["overshoot"], max_iter=params["max_iter"], p=params["p"], clip=params["clip"])
        time_end = time.time()
        total_time += time_end-time_ini
        method_iters += n_iter

    # Re-classify the perturbed image
    output = model(perturbed_data)

    # Check for success
    final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    if final_pred.item() == target.item():
      correct += 1

    else:
      incorrect += 1

  # Calculate stats
  final_acc = correct / float(iters)  # len(test_loader)
  avg_time = total_time / float(correct+incorrect)

  print("\n======== RESULTS ========")
  print("Test Accuracy = {} / {} = {:.4f}\nAverage time = {:.4f}".format(correct, iters, final_acc, avg_time))

  if method == 'deepfool':
    print("Avg. iters = {:.2f}".format(method_iters / float(correct+incorrect)))
