from src.Database_object import Database
from src.Enrichment_object import Enrichment
from src.Statistics_object import Statistics

from linear_models.Bayesian_linear_object import BayesianModel
# from logistic_models.Bayesian_logistic_object import BayesianLogisticModel


from src.Database_object import Database
from src.Statistics_object import Statistics
from linear_models.Bayesian_linear_object import BayesianModel


if __name__ == "__main__":

    # Build database
    db = Database()
    db.build_database()

    # Run statistics
    stats = Statistics(db, selector="test", export_all=True)
    stats.run_all_statistics()

    # Train Bayesian linear model
    trainer = BayesianModel(db)
    trainer.train(
        model_name="LDaq_linear",
        redo=True,
        plot_all=True,
        export_all=True,
    )

    # Final dataframe if needed
    df = db.get_dataframe