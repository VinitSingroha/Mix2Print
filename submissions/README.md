# Submissions

## 🔐 Secure Submission Process

To protect your model's predictions from being visible to other participants in the public Pull Request history, we use **End-to-End Encryption**. Only the scoring server can read your submitted file.

## Submission Structure

```
inbox/
└── YourTeamName/
    └── submission.enc
```

## How to Submit

### 1. Generate Your Predictions
Ensure your model generates a `predictions.csv` with the following columns:
- `id`: The formulation ID
- `pressure`: Predicted pressure (kPa)
- `temperature`: Predicted temperature (°C)
- `speed`: Predicted speed (mm/s)

### 2. Encrypt Your File
You must encrypt your CSV before uploading. Run the encryption tool provided in the repository:

```bash
python scripts/encrypt_submission.py predictions.csv --team YourTeamName
```
This will generate a file named `submission.enc`. **This is the only file you should upload.**

### 3. Open a Pull Request
1. Fork this repository.
2. Create your submission folder: `submissions/inbox/<YourTeamName>/`.
3. Add your `submission.enc` file to that folder.
4. Open a Pull Request from your fork to the `master` branch of the main repository.

### 4. Automatic Scoring
Once you open the PR, our bot will:
1. Securely decrypt your file using the private key stored in GitHub Secrets.
2. Calculate your NMAE score.
3. Post a comment on the PR with your results.
4. Automatically update the [Leaderboard](https://vinitsingroha.github.io/GNN-Challenge/leaderboard.html).
5. Close the Pull Request (your data is processed and recorded).

## ⚠️ One Submission Per Participant
Each GitHub user/team is allowed **exactly one** successful submission. Subsequent submissions will be automatically rejected by the scoring bot.

---
See the main [README.md](../README.md) for full competition details.
