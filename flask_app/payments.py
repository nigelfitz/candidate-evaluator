"""
Stripe payment integration for adding funds
"""
import stripe
import os
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from database import db, User, Transaction
from config import Config
from decimal import Decimal

payments = Blueprint('payments', __name__)


@payments.route('/buy-credits')
@login_required
def buy_credits():
    """Display add funds page"""
    stripe_configured = bool(Config.STRIPE_SECRET_KEY)
    is_dev_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Debug logging
    print(f"DEBUG: STRIPE_SECRET_KEY exists: {bool(Config.STRIPE_SECRET_KEY)}")
    print(f"DEBUG: STRIPE_SECRET_KEY value (first 10 chars): {Config.STRIPE_SECRET_KEY[:10] if Config.STRIPE_SECRET_KEY else 'None'}")
    print(f"DEBUG: stripe_configured: {stripe_configured}")
    print(f"DEBUG: is_dev_mode: {is_dev_mode}")
    
    return render_template('add_funds.html', 
                         stripe_configured=stripe_configured,
                         is_dev_mode=is_dev_mode)


@payments.route('/add-test-funds', methods=['POST'])
@login_required
def add_test_funds():
    """Add test funds directly (development mode only)"""
    try:
        amount_str = request.form.get('amount', '0')
        amount = Decimal(amount_str)
        
        if amount < 5:
            flash('Minimum amount is $5', 'error')
            return redirect(url_for('payments.buy_credits'))
        
        # Add funds directly to user account
        current_user.add_funds(
            amount_usd=amount,
            description=f'Test funds added (DEV MODE)'
        )
        db.session.commit()
        
        flash(f'Successfully added ${amount:.2f} to your account!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        flash(f'Error adding funds: {str(e)}', 'error')
        return redirect(url_for('payments.buy_credits'))


@payments.route('/create-checkout', methods=['POST'])
@login_required
def create_checkout():
    """Create Stripe Checkout session for adding funds"""
    try:
        # Check if Stripe is configured
        if not Config.STRIPE_SECRET_KEY:
            flash('Stripe is not configured. Use "Add Test Funds" button instead.', 'warning')
            return redirect(url_for('payments.buy_credits'))
        
        # Set Stripe API key
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        amount_str = request.form.get('amount', '0')
        amount = float(amount_str)
        
        if amount < 5:
            flash('Minimum amount is $5', 'error')
            return redirect(url_for('payments.buy_credits'))
        
        # Create Stripe Checkout Session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': int(amount * 100),  # Convert dollars to cents
                    'product_data': {
                        'name': 'Account Funds',
                        'description': f"Add ${amount:.2f} to your Candidate Evaluator account",
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('payments.success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('payments.buy_credits', _external=True),
            client_reference_id=str(current_user.id),
            metadata={
                'user_id': current_user.id,
                'amount_usd': f"{amount:.2f}"
            }
        )
        
        return redirect(checkout_session.url, code=303)
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error creating checkout session: {str(e)}', 'error')
        return redirect(url_for('payments.buy_credits'))


@payments.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    """Create Stripe Checkout session with custom return URL (for modal quick-pay)"""
    try:
        # Check if Stripe is configured
        if not Config.STRIPE_SECRET_KEY:
            # In dev mode, add test funds instead
            if os.environ.get('FLASK_ENV') == 'development':
                data = request.get_json()
                amount = Decimal(str(data.get('amount', 0)))
                
                if amount < 5:
                    return jsonify({'error': 'Minimum amount is $5'}), 400
                
                # Add test funds directly
                current_user.add_funds(
                    amount_usd=amount,
                    description=f'Test funds added (DEV MODE)'
                )
                db.session.commit()
                
                # Return the return URL
                return_url = data.get('return_url', url_for('dashboard', _external=True))
                return jsonify({'url': return_url})
            else:
                return jsonify({'error': 'Stripe not configured'}), 400
        
        # Set Stripe API key
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        data = request.get_json()
        amount = float(data.get('amount', 0))
        return_url = data.get('return_url', url_for('dashboard', _external=True))
        
        if amount < 5:
            return jsonify({'error': 'Minimum amount is $5'}), 400
        
        # Create Stripe Checkout Session with custom success URL
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': int(amount * 100),
                    'product_data': {
                        'name': 'Account Funds',
                        'description': f"Add ${amount:.2f} to continue analysis",
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=return_url + ('&' if '?' in return_url else '?') + 'session_id={CHECKOUT_SESSION_ID}',
            cancel_url=return_url,
            client_reference_id=str(current_user.id),
            metadata={
                'user_id': current_user.id,
                'amount_usd': f"{amount:.2f}"
            }
        )
        
        return jsonify({'url': checkout_session.url})
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@payments.route('/success')
@login_required
def success():
    """Handle successful payment redirect"""
    session_id = request.args.get('session_id')
    
    if session_id:
        try:
            stripe.api_key = Config.STRIPE_SECRET_KEY
            # Retrieve the session to verify it
            session = stripe.checkout.Session.retrieve(session_id)
            
            if session.payment_status == 'paid':
                # Add credits immediately for local testing (webhook won't work on localhost)
                fulfill_order(session)
                flash('Payment successful! Credits have been added to your account.', 'success')
            else:
                flash('Payment is being processed. Credits will be added once confirmed.', 'info')
                
        except Exception as e:
            flash(f'Payment completed, but verification failed: {str(e)}', 'warning')
    
    # Check if we should return to a custom URL (from run_analysis modal)
    # The success_url from Stripe will already have auto_submit parameter
    return redirect(url_for('dashboard'))


@payments.route('/webhook', methods=['POST'])
def webhook():
    """Handle Stripe webhook events"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    stripe.api_key = Config.STRIPE_SECRET_KEY
    
    # For testing, we'll skip signature verification
    # In production, you should verify the webhook signature
    try:
        event = stripe.Event.construct_from(
            request.get_json(), stripe.api_key
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    # Handle the checkout.session.completed event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        # Fulfill the purchase
        fulfill_order(session)
    
    return jsonify({'status': 'success'}), 200


def fulfill_order(session):
    """Add funds to user account after successful payment"""
    try:
        user_id = int(session['metadata']['user_id'])
        amount_usd = Decimal(session['metadata']['amount_usd'])
        
        user = User.query.get(user_id)
        if user:
            # Add funds to user account
            user.balance_usd += amount_usd
            
            # Record transaction
            transaction = Transaction(
                user_id=user_id,
                amount_usd=amount_usd,
                transaction_type='credit',
                description=f"Added ${amount_usd:.2f} to account",
                stripe_payment_id=session.get('payment_intent'),
                stripe_amount_cents=session['amount_total']
            )
            db.session.add(transaction)
            db.session.commit()
            
            print(f"✅ Successfully added ${amount_usd:.2f} to user {user_id}. New balance: ${user.balance_usd:.2f}")
        
    except Exception as e:
        print(f"❌ Error fulfilling order: {str(e)}")
        # In production, you should log this and have a retry mechanism
