"""
Step 3 - TaskRouter: decides Local vs Edge vs Cloud for each task.
================================================================================
ROMAN URDU: Yeh class faisla karti hai ke ek naya task kahan process hoga --
Local (on-device), Edge (department ka apna server), ya Cloud (shared,
door, lekin powerful). **YEH ABHI AI NAHI HAI** -- sirf ek fixed, insaan ke
likhe hue rule par mabni hai:

    compute_demand <= local_threshold                    -> Local
    local_threshold < compute_demand <= edge_threshold    -> Edge
    compute_demand > edge_threshold                       -> Cloud

Wajah: chhote kaam itne halke hote hain ke turant device par hi khatam ho
jayen (network tak jane ki zaroorat nahi), darmiyani kaam department ke
apne Edge server mein aaram se fit ho jate hain, aur sirf sab se bhaari kaam
hi Cloud ke network_delay "afford" karne ke laiq hote hain (kyunke Cloud ki
capacity bohat zyada hai). Steps 7+ mein isi jaga par ek RL agent aayega jo
seekhega ke is fixed rule se BEHTAR faisle kaise karne hain -- Step 3 ka
maqsad sirf itna hai ke teenon tiers ki queueing/capacity/delay mechanics
sahi kaam kar rahi hain, iss se pehle koi "learning" shamil ki jaye.

ENGLISH: This class decides where a new task should be processed -- Local
(on-device), Edge (the department's own server), or Cloud (shared, remote,
powerful). **THIS IS NOT AI YET** -- it is a fixed, human-written rule based
only on how big the task's compute_demand is (see the three-line rule
above). The reasoning: small jobs are cheap enough to finish instantly
on-device (no need to pay the network cost); medium jobs comfortably fit
the department's own edge server; only the biggest jobs are worth paying
the cloud's network_delay for its much larger capacity. Starting Step 7+,
an RL agent will replace this exact decision point and learn to make BETTER
choices than this fixed rule -- Step 3's only job is to prove the 3-tier
queueing/capacity/delay mechanics are correct before any learning is added
(project rule: one new idea at a time).
"""

from env.task import TIER_CLASSES


class TaskRouter:
    """Fixed (non-learned) Local/Edge/Cloud routing rule, based on a task's
    compute_demand compared against two configured thresholds."""

    def __init__(self, local_threshold: float, edge_threshold: float):
        if local_threshold <= 0:
            raise ValueError(
                f"TaskRouter: local_threshold must be > 0, got {local_threshold}"
            )
        if edge_threshold <= local_threshold:
            raise ValueError(
                f"TaskRouter: edge_threshold ({edge_threshold}) must be > "
                f"local_threshold ({local_threshold})"
            )
        self.local_threshold = local_threshold
        self.edge_threshold = edge_threshold

    def choose_tier(self, compute_demand: float) -> str:
        """Return one of TIER_CLASSES ("Local"/"Edge"/"Cloud") for a task
        with this compute_demand."""
        if compute_demand <= 0:
            raise ValueError(
                f"TaskRouter.choose_tier: compute_demand must be > 0, got {compute_demand}"
            )
        if compute_demand <= self.local_threshold:
            tier = "Local"
        elif compute_demand <= self.edge_threshold:
            tier = "Edge"
        else:
            tier = "Cloud"
        assert tier in TIER_CLASSES  # sanity check, should never fire
        return tier
