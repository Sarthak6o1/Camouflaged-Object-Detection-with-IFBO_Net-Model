model.eval()
test_loss = 0
test_mae = 0
test_salpha = 0
test_ephi = 0
test_fbw = 0
test_acc = 0
test_dice = 0
test_iou = 0

with torch.no_grad():
    for img, mask in tqdm(test_loader, desc="Evaluating on Test Set"):
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        test_loss += loss.item()
        test_mae += compute_mae(pred, mask)
        test_salpha += compute_smeasure(pred, mask)
        test_ephi += compute_ephi(pred, mask)
        test_fbw += compute_fbw(pred, mask)
        test_acc += compute_accuracy(pred, mask)
        test_dice += 1 - dice_loss(pred, mask).item()
        test_iou += iou_score(pred, mask)

n_test = len(test_loader)
print(
    f"Test Loss: {test_loss/n_test:.4f}, "
    f"Acc: {test_acc/n_test:.4f}, "
    f"MAE: {test_mae/n_test:.4f}, "
    f"Sα: {test_salpha/n_test:.4f}, "
    f"Eϕ: {test_ephi/n_test:.4f}, "
    f"Fβw: {test_fbw/n_test:.4f}, "
    f"Dice: {test_dice/n_test:.4f}, "
    f"IoU: {test_iou/n_test:.4f}"
)
