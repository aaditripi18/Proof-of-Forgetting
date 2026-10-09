"""Proof of Forgetting: auditing how federated unlearning is verified.

Pure-numpy modules (data.partition, verify.loss_mia, stats, ledger, config) import
without torch so their tests run anywhere; torch is imported only where training
or inference happens.
"""
__version__ = "0.1.0"
