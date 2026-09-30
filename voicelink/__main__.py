import os

# Everything runs from the local models/ folder; never reach for the network.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ.setdefault(_var, "1")

from .server import main  # noqa: E402

main()
