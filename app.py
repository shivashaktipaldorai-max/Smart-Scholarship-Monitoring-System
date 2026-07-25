from flask import Flask, render_template, request, redirect, url_for, flash
import csv
import os
import pandas as pd

from datetime import datetime, date

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "sih-2026-scholarship-tracker-secret-key"


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

CSV_FILE = os.path.join(
    DATA_DIR,
    "scholarship_applications.csv"
)


# ============================================================
# CSV COLUMNS
# ============================================================

FIELDS = [
    "application_id",
    "student_id",
    "student_name",
    "scheme",
    "applied_date",
    "documents_status",
    "stage",
    "sanctioned_amount",
    "disbursed_date",
    "needs_attention"
]


# ============================================================
# ALLOWED VALUES
# ============================================================

ALLOWED_DOCUMENT_STATUS = [
    "Complete",
    "Pending",
    "Incomplete"
]

ALLOWED_STAGES = [
    "Document Collection",
    "Verification",
    "Sanctioned",
    "Disbursed"
]


# ============================================================
# CREATE DATA FOLDER
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# READ APPLICATIONS
# ============================================================

def read_applications():

    applications = []

    if not os.path.exists(CSV_FILE):
        return applications

    try:

        with open(
            CSV_FILE,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if not row:
                    continue

                application_id = str(
                    row.get("application_id", "")
                ).strip()

                student_id = str(
                    row.get("student_id", "")
                ).strip()

                student_name = str(
                    row.get("student_name", "")
                ).strip()

                # Ignore duplicate header rows
                if application_id.lower() == "application_id":
                    continue

                # Ignore invalid rows
                if not application_id:
                    continue

                if not student_id:
                    continue

                if not student_name:
                    continue

                clean_row = {

                    "application_id": application_id,

                    "student_id": student_id,

                    "student_name": student_name,

                    "scheme": str(
                        row.get("scheme", "")
                    ).strip(),

                    "applied_date": str(
                        row.get("applied_date", "")
                    ).strip(),

                    "documents_status": str(
                        row.get("documents_status", "")
                    ).strip(),

                    "stage": str(
                        row.get("stage", "")
                    ).strip(),

                    "sanctioned_amount": str(
                        row.get("sanctioned_amount", "0")
                    ).strip(),

                    "disbursed_date": str(
                        row.get("disbursed_date", "")
                    ).strip(),

                    "needs_attention": str(
                        row.get("needs_attention", "")
                    ).strip()
                }

                applications.append(clean_row)

    except Exception as error:

        print(
            "CSV Read Error:",
            error
        )

    return applications


# ============================================================
# SAVE APPLICATIONS
# ============================================================

def save_applications(applications):

    try:

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        with open(
            CSV_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FIELDS
            )

            writer.writeheader()

            for application in applications:

                writer.writerow({

                    field: application.get(
                        field,
                        ""
                    )

                    for field in FIELDS

                })

        return True

    except Exception as error:

        print(
            "CSV Save Error:",
            error
        )

        return False


# ============================================================
# CALCULATE WAITING DAYS
# ============================================================

def calculate_waiting_days(application):

    try:

        applied_date_text = str(
            application.get(
                "applied_date",
                ""
            )
        ).strip()

        if not applied_date_text:
            return None

        applied_date = datetime.strptime(
            applied_date_text,
            "%Y-%m-%d"
        ).date()

        # Disbursed applications are no longer waiting
        if (
            application.get(
                "stage",
                ""
            ).strip().lower()
            == "disbursed"
        ):

            return 0

        waiting_days = (
            date.today()
            - applied_date
        ).days

        if waiting_days < 0:
            return 0

        return waiting_days

    except Exception as error:

        print(
            "Waiting Days Error:",
            error
        )

        return None


# ============================================================
# MACHINE LEARNING MODEL
#
# Target:
# needs_attention
#
# 0 = No immediate attention
# 1 = Needs attention
#
# Features:
# - documents_status
# - stage
# - sanctioned_amount
# ============================================================

def train_model():

    if not os.path.exists(CSV_FILE):

        print(
            "ML Error: CSV file not found."
        )

        return None

    try:

        # Read CSV
        data = pd.read_csv(
            CSV_FILE
        )

        # ----------------------------------------------------
        # CLEAN COLUMN NAMES
        # ----------------------------------------------------

        data.columns = (
            data.columns
            .astype(str)
            .str.strip()
        )

        # ----------------------------------------------------
        # CHECK REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [

            "application_id",

            "documents_status",

            "stage",

            "sanctioned_amount",

            "needs_attention"

        ]

        for column in required_columns:

            if column not in data.columns:

                print(
                    f"ML Error: Missing column '{column}'"
                )

                return None

        # ----------------------------------------------------
        # REMOVE DUPLICATE HEADER ROWS
        # ----------------------------------------------------

        data = data[
            data["application_id"].notna()
        ]

        data = data[
            data["application_id"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "application_id"
        ]

        # ----------------------------------------------------
        # CLEAN DOCUMENT STATUS
        # ----------------------------------------------------

        data["documents_status"] = (

            data["documents_status"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()

        )

        # ----------------------------------------------------
        # CLEAN STAGE
        # ----------------------------------------------------

        data["stage"] = (

            data["stage"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()

        )

        # ----------------------------------------------------
        # CLEAN SANCTIONED AMOUNT
        # ----------------------------------------------------

        data["sanctioned_amount"] = (

            pd.to_numeric(

                data["sanctioned_amount"],

                errors="coerce"

            )

            .fillna(0)

        )

        # ----------------------------------------------------
        # CLEAN TARGET
        # ----------------------------------------------------

        data["needs_attention"] = (
            data["needs_attention"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        target_mapping = {

            "yes": 1,

            "no": 0,

            "1": 1,

            "0": 0,

            "true": 1,

            "false": 0,

            "needs attention": 1,

            "no immediate attention needed": 0

        }


        data["needs_attention"] = (
            data["needs_attention"]
            .map(target_mapping)
        )


        # Remove invalid target values

        data = data.dropna(
            subset=[
                "needs_attention"
            ]
        )


        data["needs_attention"] = (
            data["needs_attention"]
            .astype(int)
        )


        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        print("\n========== ML DATA CHECK ==========")

        print(
            "Valid Training Records:",
            len(data)
        )

        print(
            "\nAttention Label Distribution:"
        )

        print(
            data["needs_attention"]
            .value_counts()
        )


        print(
            "\nDocument Status Values:"
        )

        print(
            data["documents_status"]
            .unique()
        )


        print(
            "\nStage Values:"
        )

        print(
            data["stage"]
            .unique()
        )

        print(
            "===================================\n"
        )

        # ----------------------------------------------------
        # CHECK MINIMUM DATA
        # ----------------------------------------------------

        if len(data) < 10:

            print(
                "ML Error: At least 10 valid records are required."
            )

            return None

        # ----------------------------------------------------
        # CHECK BOTH CLASSES
        # ----------------------------------------------------

        if data["needs_attention"].nunique() < 2:

            print(
                "ML Error: Both 0 and 1 classes are required."
            )

            return None

        # ----------------------------------------------------
        # ENCODE DOCUMENT STATUS
        # ----------------------------------------------------

        documents_encoder = LabelEncoder()

        data["documents_status_encoded"] = (

            documents_encoder.fit_transform(

                data["documents_status"]

            )

        )

        # ----------------------------------------------------
        # ENCODE STAGE
        # ----------------------------------------------------

        stage_encoder = LabelEncoder()

        data["stage_encoded"] = (

            stage_encoder.fit_transform(

                data["stage"]

            )

        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        X = data[

            [

                "documents_status_encoded",

                "stage_encoded",

                "sanctioned_amount"

            ]

        ]

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        y = data[
            "needs_attention"
        ].astype(int)

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

        # ----------------------------------------------------
        # DECISION TREE MODEL
        # ----------------------------------------------------

        model = DecisionTreeClassifier(

            random_state=42,

            max_depth=4,

            min_samples_leaf=2

        )

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        model.fit(

            X_train,

            y_train

        )

        # ----------------------------------------------------
        # TEST MODEL
        # ----------------------------------------------------

        y_pred = model.predict(

            X_test

        )

        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        accuracy = accuracy_score(

            y_test,

            y_pred

        )

        print(
            "\n===================================="
        )

        print(
            "SCHOLARSHIP DELAY RISK ML MODEL"
        )

        print(
            "===================================="
        )

        print(
            "Training Records:",
            len(X_train)
        )

        print(
            "Testing Records:",
            len(X_test)
        )

        print(
            "Accuracy:",
            round(
                accuracy * 100,
                2
            ),
            "%"
        )

        print(
            "\nClassification Report:"
        )

        print(

            classification_report(

                y_test,

                y_pred,

                zero_division=0

            )

        )

        print(
            "====================================\n"
        )

        return {

            "model": model,

            "documents_encoder":
                documents_encoder,

            "stage_encoder":
                stage_encoder,

            "accuracy":
                accuracy

        }

    except Exception as error:

        print(
            "ML Training Error:",
            error
        )

        return None


# ============================================================
# PREDICT APPLICATION DELAY RISK
# ============================================================

def predict_application(

    documents_status,

    stage,

    sanctioned_amount

):

    try:

        # Train model
        result = train_model()

        # If model cannot be trained
        if result is None:

            return {

                "status":
                    "unavailable",

                "message":
                    "Prediction Unavailable",

                "confidence":
                    0,

                "risk":
                    "Unknown"

            }

        # Get trained objects
        model = result[
            "model"
        ]

        documents_encoder = result[
            "documents_encoder"
        ]

        stage_encoder = result[
            "stage_encoder"
        ]

        # Normalize input
        documents_status = str(
            documents_status
        ).strip()

        stage = str(
            stage
        ).strip()

        # ----------------------------------------------------
        # CHECK UNKNOWN DOCUMENT STATUS
        # ----------------------------------------------------

        if documents_status not in documents_encoder.classes_:

            return {

                "status":
                    "unavailable",

                "message":
                    "Prediction Unavailable",

                "confidence":
                    0,

                "risk":
                    "Unknown"

            }

        # ----------------------------------------------------
        # CHECK UNKNOWN STAGE
        # ----------------------------------------------------

        if stage not in stage_encoder.classes_:

            return {

                "status":
                    "unavailable",

                "message":
                    "Prediction Unavailable",

                "confidence":
                    0,

                "risk":
                    "Unknown"

            }

        # ----------------------------------------------------
        # ENCODE DOCUMENT STATUS
        # ----------------------------------------------------

        documents_encoded = (

            documents_encoder.transform(

                [
                    documents_status
                ]

            )[0]

        )

        # ----------------------------------------------------
        # ENCODE STAGE
        # ----------------------------------------------------

        stage_encoded = (

            stage_encoder.transform(

                [
                    stage
                ]

            )[0]

        )

        # ----------------------------------------------------
        # CONVERT AMOUNT
        # ----------------------------------------------------

        try:

            amount = float(

                sanctioned_amount

                or 0

            )

        except Exception:

            amount = 0

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame(

            [[

                documents_encoded,

                stage_encoded,

                amount

            ]],

            columns=[

                "documents_status_encoded",

                "stage_encoded",

                "sanctioned_amount"

            ]

        )

        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(

            input_data

        )[0]

        # ----------------------------------------------------
        # GET PROBABILITY
        # ----------------------------------------------------

        probabilities = model.predict_proba(

            input_data

        )[0]

        # Find probability of predicted class
        predicted_class_index = list(
            model.classes_
        ).index(
            prediction
        )

        confidence = (

            probabilities[
                predicted_class_index
            ]

            * 100

        )

        # ----------------------------------------------------
        # CONVERT PREDICTION TO TEXT
        # ----------------------------------------------------

        if int(prediction) == 1:

            return {

                "status":
                    "success",

                "message":
                    "Needs Attention",

                "confidence":
                    round(
                        confidence,
                        2
                    ),

                "risk":
                    "High"

            }

        else:

            return {

                "status":
                    "success",

                "message":
                    "No Immediate Attention Needed",

                "confidence":
                    round(
                        confidence,
                        2
                    ),

                "risk":
                    "Low"

            }

    except Exception as error:

        print(
            "Prediction Error:",
            error
        )

        return {

            "status":
                "unavailable",

            "message":
                "Prediction Unavailable",

            "confidence":
                0,

            "risk":
                "Unknown"

        }


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    applications = read_applications()

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search = request.args.get(
        "search",
        ""
    ).strip().lower()

    if search:

        filtered_applications = []

        for application in applications:

            searchable_text = " ".join([

                application.get(
                    "application_id",
                    ""
                ),

                application.get(
                    "student_id",
                    ""
                ),

                application.get(
                    "student_name",
                    ""
                ),

                application.get(
                    "scheme",
                    ""
                )

            ]).lower()

            if search in searchable_text:

                filtered_applications.append(
                    application
                )

        applications = filtered_applications

    # --------------------------------------------------------
    # ADD WAITING DAYS AND ML PREDICTION
    # --------------------------------------------------------

    for application in applications:

        # Calculate waiting days
        waiting_days = calculate_waiting_days(
            application
        )

        application[
            "waiting_days"
        ] = waiting_days

        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        prediction_result = predict_application(

            application.get(
                "documents_status",
                ""
            ),

            application.get(
                "stage",
                ""
            ),

            application.get(
                "sanctioned_amount",
                0
            )

        )

        application[
            "prediction"
        ] = prediction_result[
            "message"
        ]

        application[
            "prediction_confidence"
        ] = prediction_result[
            "confidence"
        ]

        application[
            "risk"
        ] = prediction_result[
            "risk"
        ]

    # --------------------------------------------------------
    # LONGEST WAITING APPLICATIONS
    # --------------------------------------------------------

    applications_with_waiting = [

        application

        for application in applications

        if application.get(
            "waiting_days"
        ) is not None

    ]

    longest_waiting = sorted(

        applications_with_waiting,

        key=lambda x: x.get(
            "waiting_days",
            0
        ),

        reverse=True

    )[:5]

    # --------------------------------------------------------
    # TOTAL APPLICATIONS
    # --------------------------------------------------------

    total_applications = len(
        applications
    )

    # --------------------------------------------------------
    # LONGEST WAITING DAYS
    # --------------------------------------------------------

    if applications_with_waiting:

        longest_waiting_days = max(

            application.get(
                "waiting_days",
                0
            )

            for application
            in applications_with_waiting

        )

    else:

        longest_waiting_days = 0

    # --------------------------------------------------------
    # ATTENTION REQUIRED
    # --------------------------------------------------------

    attention_required = sum(

        1

        for application
        in applications

        if application.get(
            "prediction"
        )
        == "Needs Attention"

    )

    # --------------------------------------------------------
    # TOTAL SANCTIONED AMOUNT
    # --------------------------------------------------------

    total_sanctioned = 0

    for application in applications:

        try:

            total_sanctioned += float(

                application.get(
                    "sanctioned_amount",
                    0
                )

                or 0

            )

        except Exception:

            pass

    # --------------------------------------------------------
    # RENDER PAGE
    # --------------------------------------------------------

    return render_template(

        "index.html",

        applications=applications,

        longest_waiting=longest_waiting,

        total_applications=total_applications,

        longest_waiting_days=longest_waiting_days,

        attention_required=attention_required,

        total_sanctioned=total_sanctioned,

        search=search

    )


# ============================================================
# ADD NEW APPLICATION
# ============================================================

@app.route(
    "/add",
    methods=["POST"]
)
def add_application():

    applications = read_applications()

    # --------------------------------------------------------
    # GET FORM DATA
    # --------------------------------------------------------

    application_id = request.form.get(
        "application_id",
        ""
    ).strip()

    student_id = request.form.get(
        "student_id",
        ""
    ).strip()

    student_name = request.form.get(
        "student_name",
        ""
    ).strip()

    scheme = request.form.get(
        "scheme",
        ""
    ).strip()

    applied_date = request.form.get(
        "applied_date",
        ""
    ).strip()

    documents_status = request.form.get(
        "documents_status",
        ""
    ).strip()

    stage = request.form.get(
        "stage",
        ""
    ).strip()

    sanctioned_amount = request.form.get(
        "sanctioned_amount",
        "0"
    ).strip()

    disbursed_date = request.form.get(
        "disbursed_date",
        ""
    ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not application_id:

        flash(
            "Application ID is required."
        )

        return redirect(
            url_for("index")
        )

    if not student_id:

        flash(
            "Student ID is required."
        )

        return redirect(
            url_for("index")
        )

    if not student_name:

        flash(
            "Student Name is required."
        )

        return redirect(
            url_for("index")
        )

    if not scheme:

        flash(
            "Scholarship Scheme is required."
        )

        return redirect(
            url_for("index")
        )

    if not applied_date:

        flash(
            "Applied Date is required."
        )

        return redirect(
            url_for("index")
        )

    if documents_status not in ALLOWED_DOCUMENT_STATUS:

        flash(
            "Invalid Documents Status."
        )

        return redirect(
            url_for("index")
        )

    if stage not in ALLOWED_STAGES:

        flash(
            "Invalid Application Stage."
        )

        return redirect(
            url_for("index")
        )

    # --------------------------------------------------------
    # CHECK DUPLICATE APPLICATION ID
    # --------------------------------------------------------

    for application in applications:

        if application.get(
            "application_id",
            ""
        ).lower() == application_id.lower():

            flash(
                "Application ID already exists."
            )

            return redirect(
                url_for("index")
            )

    # --------------------------------------------------------
    # VALIDATE AMOUNT
    # --------------------------------------------------------

    try:

        amount = float(
            sanctioned_amount
            or 0
        )

        if amount < 0:

            raise ValueError

    except Exception:

        flash(
            "Invalid sanctioned amount."
        )

        return redirect(
            url_for("index")
        )

    # --------------------------------------------------------
    # AUTOMATIC ATTENTION LABEL
    #
    # This label is used as the training target.
    #
    # Rules:
    # - Pending/Incomplete documents = Attention
    # - Very old waiting application = Attention
    # - Otherwise = No immediate attention
    # --------------------------------------------------------

    new_application = {

        "application_id":
            application_id,

        "student_id":
            student_id,

        "student_name":
            student_name,

        "scheme":
            scheme,

        "applied_date":
            applied_date,

        "documents_status":
            documents_status,

        "stage":
            stage,

        "sanctioned_amount":
            str(amount),

        "disbursed_date":
            disbursed_date,

        "needs_attention":
            "0"

    }

    # Calculate current waiting days
    waiting_days = calculate_waiting_days(
        new_application
    )

    # Determine attention status
    if documents_status in [
        "Pending",
        "Incomplete"
    ]:

        new_application[
            "needs_attention"
        ] = "1"

    elif (

        waiting_days is not None

        and waiting_days >= 30

        and stage != "Disbursed"

    ):

        new_application[
            "needs_attention"
        ] = "1"

    else:

        new_application[
            "needs_attention"
        ] = "0"

    # --------------------------------------------------------
    # ADD APPLICATION
    # --------------------------------------------------------

    applications.append(
        new_application
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    if save_applications(
        applications
    ):

        flash(
            "Application added successfully."
        )

    else:

        flash(
            "Error saving application."
        )

    return redirect(
        url_for("index")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "SMART SCHOLARSHIP MONITORING SYSTEM"
    )

    print(
        "===================================="
    )

    print(
        "CSV File:",
        CSV_FILE
    )

    print(
        "Server running at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "====================================\n"
    )

    app.run(
        debug=True
    )