import matplotlib.pyplot as plt

from experiments.radiative_closure_single_date.analysis import (
    plot_albedo_cloud_coupling_diagnostics,
    plot_albedo_cloud_coupling_residuals,
    plot_albedo_kernel_comparison,
    plot_albedo_response_stack,
    plot_albedo_sensitivity_change,
    plot_feedback_ablation_residuals,
    plot_feedback_component_responses,
    plot_feedback_total_response,
    plot_neurips_albedo_sensitivity,
    setup_neurips_feedback_context,
)
from experiments.standard_eval.analysis import (
    write_neurips_summary_tables,
    write_sobolev_lambda_summary_tables,
)


def save(fig):
    plt.close(fig)


def make_feedback_figures() -> None:
    setup_neurips_feedback_context()
    for arctic in [False, True]:
        for sky in ["all", "clr"]:
            save(plot_feedback_total_response(sky, arctic=arctic))
            save(plot_feedback_component_responses(sky, arctic=arctic))

    save(plot_feedback_ablation_residuals(arctic=True))
    save(plot_albedo_cloud_coupling_diagnostics(arctic=True))
    save(plot_albedo_cloud_coupling_residuals(arctic=True))
    save(plot_albedo_response_stack(arctic=True))
    save(
        plot_albedo_kernel_comparison(
            base_date="2012-08", perturbed_date="2015-08", arctic=True
        )
    )
    save(
        plot_albedo_sensitivity_change(
            base_date="2012-08", perturbed_date="2015-08", arctic=True
        )
    )
    save(plot_neurips_albedo_sensitivity("2015-07", arctic=False))


def main() -> None:
    write_neurips_summary_tables()
    write_sobolev_lambda_summary_tables()
    make_feedback_figures()


if __name__ == "__main__":
    main()
