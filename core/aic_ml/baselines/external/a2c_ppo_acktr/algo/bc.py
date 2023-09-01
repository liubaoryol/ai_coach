import torch
import torch.optim as optim


class BC():

  def __init__(self, actor_critic, entropy_coef=1e-3, lr=None, eps=None):

    self.actor_critic = actor_critic
    self.entropy_coef = entropy_coef

    self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

  def train(self, gail_train_loader, device, use_confidence=False):
    all_loss = []
    L2_WEIGHT = 0.0
    for expert_batch in gail_train_loader:
      if use_confidence:
        expert_state, expert_action, confidence = expert_batch
        confidence = confidence.to(device=device)
      else:
        expert_state, expert_action = expert_batch
        confidence = 1.0

      expert_state = torch.as_tensor(expert_state, device=device)
      expert_action = torch.as_tensor(expert_action, device=device)

      _, log_prob, dist_entropy, _ = self.actor_critic.evaluate_actions(
          expert_state, None, None, expert_action)
      log_prob = confidence * log_prob

      log_prob = log_prob.mean()
      dist_entropy = dist_entropy.mean()

      l2_norms = [
          torch.sum(torch.square(w)) for w in self.actor_critic.parameters()
      ]
      l2_norm = sum(l2_norms) / 2

      ent_loss = -self.entropy_coef * dist_entropy
      neglogp = -log_prob
      l2_loss = L2_WEIGHT * l2_norm
      loss = neglogp + ent_loss + l2_loss

      all_loss.append(loss.item())

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    return sum(all_loss) / len(all_loss)
