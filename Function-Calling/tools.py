# ==========================================
# tools.py: Multi-Tool Registry for Module 11
# ==========================================

# 1. Raw Python Function Implementations (The actual logic)

def get_student_grade(student_name: str, subject: str) -> str:
    """Retrieves the academic grade of a student for a specific subject."""
    database = {
        "Sreejith": {"Data Structures": "A+", "Java": "A", "Python": "O", "SQL": "A+"},
        "Priya": {"Data Structures": "B+", "Java": "A+", "Python": "A"}
    }
    student = student_name.capitalize()
    courses = database.get(student, {})
    grade = courses.get(subject, "Subject or student record not found.")
    return f"{student}'s grade in {subject} is {grade}."


def get_assignment_deadline(subject: str) -> str:
    """Checks the upcoming deadline for a specific subject assignment."""
    deadlines = {
        "Data Structures": "November 5, 2026",
        "Java": "November 10, 2026",
        "Python": "November 3, 2026",
        "SQL": "November 12, 2026"
    }
    deadline = deadlines.get(subject, "No active assignment deadlines found for this subject.")
    return f"The upcoming deadline for {subject} is {deadline}."


def get_course_instructor(subject: str) -> str:
    """Finds the faculty member teaching a specific course."""
    faculty = {
        "Data Structures": "Dr. Ramesh Kumar",
        "Java": "Prof. Anand Sundaram",
        "Python": "Dr. Priya Dharshini",
        "SQL": "Prof. Karthikayan"
    }
    instructor = faculty.get(subject, "Instructor information not available.")
    return f"The instructor for {subject} is {instructor}."


# 2. OpenAI-Compatible Tool Schemas (The definitions the LLM reads)

student_tools_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_student_grade",
            "description": "Retrieves the academic grade of a student for a specific subject.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_name": {
                        "type": "string",
                        "description": "The name of the student (e.g., Sreejith)"
                    },
                    "subject": {
                        "type": "string",
                        "description": "The name of the course (e.g., Data Structures, Java, Python, SQL)"
                    }
                },
                "required": ["student_name", "subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_assignment_deadline",
            "description": "Checks the upcoming submission deadline for a course assignment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The name of the subject (e.g., Data Structures, Java, Python, SQL)"
                    }
                },
                "required": ["subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_course_instructor",
            "description": "Finds the faculty instructor teaching a specific course.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The name of the subject (e.g., Data Structures, Java, Python, SQL)"
                    }
                },
                "required": ["subject"]
            }
        }
    }
]


# 3. Execution Dispatcher Mapping
# This dictionary maps the string name of the function to the actual Python function object,
# making it easy to execute whatever tool the LLM requests dynamically.

TOOL_REGISTRY = {
    "get_student_grade": get_student_grade,
    "get_assignment_deadline": get_assignment_deadline,
    "get_course_instructor": get_course_instructor,
}