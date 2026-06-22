from django.db import models
from django.utils import timezone
from django.db import models
from django.contrib.auth.models import User
from accounts.models import StaffProfile, Student, SystemConfiguration
from dashboard.models import Routine


class AttendanceSession(models.Model):
    subject_name = models.CharField(max_length=100)
    start_time = models.DateTimeField()  # REMOVED auto_now_add=True
    expected_duration = models.PositiveIntegerField(default=60)  # in minutes
    routine = models.ForeignKey(Routine, on_delete=models.SET_NULL, null=True, related_name="sessions")
    date = models.DateField()  # REMOVED auto_now_add=True
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(StaffProfile, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)  # Optional: track when session was created

    def __str__(self):
        return f"{self.subject_name} ({self.date})"


class AttendanceLog(models.Model):
    STATUS_CHOICES = [
        ("PRESENT", "Present"),
        ("ABSENT", "Absent"),
        ("LATE", "Late"),
        ("LEAVE", "Authorized Leave"),
    ]

    session = models.ForeignKey(
        AttendanceSession, on_delete=models.CASCADE, related_name="logs"
    )
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    first_seen = models.DateTimeField(default=timezone.now) 
    last_seen = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="ABSENT")
    is_manual = models.BooleanField(default=False)
    confidence = models.FloatField(null=True, blank=True)

    @property
    def presence_duration_minutes(self):
        """Calculate presence duration in minutes"""
        if self.first_seen and self.last_seen:
            duration = (self.last_seen - self.first_seen).total_seconds() / 60
            return duration
        return 0

    @property
    def retention_percentage(self):
        if self.session.expected_duration <= 0:
            return 0
        return (self.presence_duration_minutes / self.session.expected_duration) * 100

    def save(self, *args, **kwargs):
        # 1. Update last_seen if not manual
        if not self.is_manual:
            self.last_seen = timezone.now()

        # 2. Get the global config
        config = SystemConfiguration.load()
        
        # 3. Use the dynamic threshold from configuration
        if self.retention_percentage >= config.min_retention_required:
            self.status = "PRESENT"
        elif self.presence_duration_minutes > 5:  # Small buffer for 'Late'
            self.status = "LATE"
        else:
            self.status = "ABSENT"

        super().save(*args, **kwargs)
        
    class Meta:
        unique_together = ("session", "student")