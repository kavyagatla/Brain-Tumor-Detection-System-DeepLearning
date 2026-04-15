from sqlalchemy import func


@main.route('/dashboard')
@login_required
def dashboard():
    # 1. Fetch User History
    history = Scan.query.filter_by(doctor_id=current_user.id).order_by(Scan.upload_date.desc()).all()

    # 2. Calculate Statistics for Charts
    total_scans = len(history)

    # Get Tumor Distribution (e.g., {'Meningioma': 4, 'Glioma': 2})
    tumor_counts = db.session.query(
        Scan.tumor_type, func.count(Scan.tumor_type)
    ).filter_by(doctor_id=current_user.id).group_by(Scan.tumor_type).all()

    # Convert to format for Chart.js
    labels = [t[0] for t in tumor_counts]
    data = [t[1] for t in tumor_counts]

    return render_template('dashboard/dashboard.html',
                           history=history,
                           total_scans=total_scans,
                           chart_labels=labels,
                           chart_data=data)