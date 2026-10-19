# NUC user services

Copies of the units installed in `~/.config/systemd/user/`. They start at boot
(the `deploy` user has linger enabled), resume from the checkpoints, restart
after a crash or out-of-memory kill (at most 3 times in 6 h), and skip
themselves once their work is complete (`scripts/nuc_todo.py`).

| Unit | Job | Log |
|---|---|---|
| `chemoqbit-tzvp` | def2-TZVP single points (resumable within a structure) | `results/service_tzvp.log` |
| `chemoqbit-xc` | functional check of the embedded fragment, 3 GB cap | `results/service_xc.log` |
| `chemoqbit-notify` | mails step completions (needs `~/.config/chemoqbit/smtp.ini`) | `results/notify_watch.log` |

Install: `cp systemd/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload &&
systemctl --user enable --now chemoqbit-tzvp chemoqbit-xc chemoqbit-notify`

Remove: `systemctl --user disable --now chemoqbit-tzvp chemoqbit-xc chemoqbit-notify`,
then delete the files.
